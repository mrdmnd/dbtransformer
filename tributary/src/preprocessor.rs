//! Preprocessor binary: transforms raw parquet databases into graph representation.
//!
//! ## Output Format
//!
//! The preprocessor outputs a directory per database:
//!
//! ```text
//! database_name/
//!   manifest.json       # Metadata, stats, file paths
//!   schema.rkyv         # Tables, columns, column embeddings
//!   graph.rkyv          # CSR adjacency (outgoing + incoming)
//!   cells.rkyv          # Cell values, row offsets, row timestamps
//!   embeddings.bin      # Text embeddings (raw f16, mmap'd)
//! ```
//!
//! ## Streaming Architecture
//!
//! Memory usage is O(chunk_size) regardless of dataset size:
//!
//! 1. **Schema Discovery** - Collect column types and normalization statistics
//! 2. **Vocabulary Extraction** - Stream text columns, deduplicate with HashSet, write to disk
//! 3. **Embedding** - Stream vocab file through embedder, write directly to `embeddings.bin`
//! 4. **Cell Encoding** - Stream tables, encode cells with vocab lookup, build PK index
//! 5. **FK Edge Building** - Stream FK columns, resolve to row indices, build CSR graph
//! 6. **Finalize** - Save schema, graph, cells to separate .rkyv files
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --bin preprocessor -- \
//!   --input-dir databases_raw/rel-event \
//!   --output-dir databases_preprocessed/ \
//!   --chunk-size 100000 \
//!   --verbose
//! ```
//!
//! ## Loading
//!
//! Use `Database::load()` to load the preprocessed database from a directory.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use tracing::{Level, debug, info, warn};
use tracing_subscriber::FmtSubscriber;

use tributary::{
    Cells, Column as SchemaColumn, ColumnIdx, DatabaseMetadata, Embedder, EmbedderConfig,
    EmbeddingIdx, Graph, Manifest, ManifestStats, NO_TIMESTAMP, PackedCell, RowIdx, Schema,
    SemanticType, Table, TableIdx, TableMetadata, load_metadata, pack_embedding_idx, pack_null,
    pack_numerical, pack_timestamp,
};

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "preprocessor")]
#[command(about = "Preprocess a database from parquet files into graph representation.")]
struct Args {
    #[arg(short, long)]
    input_dir: PathBuf,

    #[arg(short, long, default_value = ".")]
    output_dir: PathBuf,

    #[arg(short, long, default_value = "false")]
    verbose: bool,

    /// Number of rows to process at a time (controls memory usage).
    #[arg(long, default_value = "100000")]
    chunk_size: usize,
}

// ============================================================================
// Streaming Vocabulary
// ============================================================================

struct StreamingVocab {
    temp_dir: PathBuf,
    vocab_path: PathBuf,
    pub embeddings_path: PathBuf,
    lookup: HashMap<u64, EmbeddingIdx>,
    pub vocab_size: usize,
    pub embed_dim: usize,
}

impl StreamingVocab {
    fn new(output_dir: &PathBuf, db_name: &str) -> Result<Self> {
        let temp_dir = output_dir.join(format!(".{}_temp", db_name));
        std::fs::create_dir_all(&temp_dir)?;

        Ok(Self {
            vocab_path: temp_dir.join("vocab.txt"),
            embeddings_path: output_dir.join(db_name).join("embeddings.bin"),
            temp_dir,
            lookup: HashMap::new(),
            vocab_size: 0,
            embed_dim: 0,
        })
    }

    fn hash_text(text: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    fn extract_vocabulary(
        &mut self,
        table_infos: &[TableInfo],
        schema: &Schema,
        chunk_size: usize,
    ) -> Result<usize> {
        info!("Extracting vocabulary from text columns...");

        let mut unique_texts: HashSet<String> = HashSet::new();

        let total_rows: usize = table_infos.iter().map(|t| t.num_rows).sum();
        let pb = ProgressBar::new(total_rows as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} rows",
            )
            .unwrap()
            .progress_chars("█▓▒░  "),
        );

        let mut rows_processed = 0u64;

        for (table_idx, info) in table_infos.iter().enumerate() {
            let table_idx = TableIdx(table_idx as u32);
            let path_str = path_to_str(&info.path)?;

            let ignored_cols: Vec<&str> = info
                .metadata
                .as_ref()
                .map(|m| m.ignored_columns.iter().map(|s| s.as_str()).collect())
                .unwrap_or_default();

            let text_columns: Vec<(String, SemanticType)> = info
                .column_stats
                .iter()
                .filter(|cs| {
                    !should_ignore_column(&cs.name, &ignored_cols)
                        && (cs.stype == SemanticType::Text || cs.stype == SemanticType::Categorical)
                })
                .map(|cs| (cs.name.clone(), cs.stype))
                .collect();

            if text_columns.is_empty() {
                rows_processed += info.num_rows as u64;
                pb.set_position(rows_processed);
                continue;
            }

            let fk_columns: HashSet<String> = info
                .metadata
                .as_ref()
                .map(|m| {
                    m.foreign_key_column_to_primary_key_table
                        .keys()
                        .cloned()
                        .collect()
                })
                .unwrap_or_default();

            let mut offset: i64 = 0;
            let chunk_size_u32 = chunk_size as u32;

            while (offset as usize) < info.num_rows {
                let df =
                    LazyFrame::scan_parquet(PlPath::new(path_str), ScanArgsParquet::default())?
                        .slice(offset, chunk_size_u32)
                        .collect()
                        .with_context(|| format!("Failed to read chunk from: {:?}", info.path))?;

                let batch_rows = df.height();
                if batch_rows == 0 {
                    break;
                }

                for (col_name, stype) in &text_columns {
                    if fk_columns.contains(col_name) {
                        continue;
                    }
                    let is_pk = schema.tables[table_idx.0 as usize]
                        .primary_key_column
                        .map(|idx| &schema.columns[idx.0 as usize].name == col_name)
                        .unwrap_or(false);
                    if is_pk {
                        continue;
                    }

                    if let Ok(col) = df.column(col_name) {
                        extract_texts_from_column(col, col_name, *stype, &mut unique_texts);
                    }
                }

                rows_processed += batch_rows as u64;
                offset += batch_rows as i64;
                pb.set_position(rows_processed);
            }
        }

        pb.finish_with_message("Vocabulary extracted");

        info!("Sorting {} unique texts...", unique_texts.len());
        let mut texts: Vec<String> = unique_texts.into_iter().collect();
        texts.sort();

        info!("Writing vocabulary to file...");
        let write_pb = ProgressBar::new(texts.len() as u64);
        write_pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} texts",
            )
            .unwrap()
            .progress_chars("█▓▒░  "),
        );

        let vocab_file = File::create(&self.vocab_path)?;
        let mut writer = BufWriter::new(vocab_file);
        for (idx, text) in texts.iter().enumerate() {
            writeln!(writer, "{}", text)?;
            if idx % 100_000 == 0 {
                write_pb.set_position(idx as u64);
            }
        }
        writer.flush()?;
        write_pb.finish_with_message("Vocabulary written");

        self.vocab_size = texts.len();

        info!("Building lookup table...");
        let lookup_pb = ProgressBar::new(texts.len() as u64);
        lookup_pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} hashes",
            )
            .unwrap()
            .progress_chars("█▓▒░  "),
        );

        self.lookup.reserve(texts.len());
        for (idx, text) in texts.iter().enumerate() {
            let hash = Self::hash_text(text);
            self.lookup.insert(hash, EmbeddingIdx(idx as u32));
            if idx % 100_000 == 0 {
                lookup_pb.set_position(idx as u64);
            }
        }
        lookup_pb.finish_with_message("Lookup table built");

        info!("Vocabulary complete: {} unique texts", self.vocab_size);
        Ok(self.vocab_size)
    }

    fn embed_vocabulary(&mut self, embedder: &Embedder) -> Result<()> {
        if self.vocab_size == 0 {
            // Create empty embeddings file with header
            std::fs::create_dir_all(self.embeddings_path.parent().unwrap())?;
            let embed_file = File::create(&self.embeddings_path)?;
            let mut writer = BufWriter::new(embed_file);
            writer.write_all(&0u32.to_le_bytes())?;
            writer.write_all(&(embedder.embedding_dim() as u32).to_le_bytes())?;
            writer.flush()?;
            self.embed_dim = embedder.embedding_dim();
            return Ok(());
        }

        info!(
            "Embedding {} texts (batch_size={})...",
            self.vocab_size, embedder.config.batch_size
        );

        self.embed_dim = embedder.embedding_dim();
        let batch_size = embedder.config.batch_size;

        std::fs::create_dir_all(self.embeddings_path.parent().unwrap())?;
        let embed_file = File::create(&self.embeddings_path)?;
        let mut writer = BufWriter::new(embed_file);

        writer.write_all(&(self.vocab_size as u32).to_le_bytes())?;
        writer.write_all(&(self.embed_dim as u32).to_le_bytes())?;

        let vocab_file = File::open(&self.vocab_path)?;
        let reader = BufReader::new(vocab_file);
        let mut batch: Vec<String> = Vec::with_capacity(batch_size);
        let mut texts_embedded = 0usize;

        let pb = ProgressBar::new(self.vocab_size as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} Embedding: [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, ETA: {eta})",
            )
            .unwrap()
            .progress_chars("█▓▒░  "),
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(100));

        let total_batches = (self.vocab_size + batch_size - 1) / batch_size;
        let log_interval = std::cmp::max(1, total_batches / 20);
        let mut batches_processed = 0usize;

        for line in reader.lines() {
            let text = line?;
            batch.push(text);

            if batch.len() >= batch_size {
                self.embed_and_write_batch(&batch, embedder, &mut writer)?;
                texts_embedded += batch.len();
                batches_processed += 1;
                pb.set_position(texts_embedded as u64);

                if batches_processed % log_interval == 0 {
                    let pct = (texts_embedded as f64 / self.vocab_size as f64) * 100.0;
                    info!(
                        "Embedding progress: {}/{} texts ({:.1}%)",
                        texts_embedded, self.vocab_size, pct
                    );
                }
                batch.clear();
            }
        }

        if !batch.is_empty() {
            self.embed_and_write_batch(&batch, embedder, &mut writer)?;
            texts_embedded += batch.len();
            pb.set_position(texts_embedded as u64);
        }

        writer.flush()?;
        pb.finish_with_message("Embeddings complete");

        info!(
            "Embedded {} texts to {:?}",
            texts_embedded, self.embeddings_path
        );
        Ok(())
    }

    fn embed_and_write_batch(
        &self,
        batch: &[String],
        embedder: &Embedder,
        writer: &mut BufWriter<File>,
    ) -> Result<()> {
        let refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();

        match embedder.embed_batch_f16(&refs) {
            Ok(embeddings) => {
                for embedding in embeddings {
                    for val in embedding {
                        writer.write_all(&val.to_le_bytes())?;
                    }
                }
            }
            Err(e) => {
                warn!("Failed to embed batch: {}", e);
                let zeros = vec![0u8; self.embed_dim * 2];
                for _ in batch {
                    writer.write_all(&zeros)?;
                }
            }
        }
        Ok(())
    }

    fn lookup_text(&self, text: &str) -> Option<EmbeddingIdx> {
        let hash = Self::hash_text(text);
        self.lookup.get(&hash).copied()
    }

    fn cleanup(&self) -> Result<()> {
        if self.temp_dir.exists() {
            std::fs::remove_dir_all(&self.temp_dir)?;
        }
        Ok(())
    }
}

fn extract_texts_from_column(
    col: &Column,
    col_name: &str,
    stype: SemanticType,
    texts: &mut HashSet<String>,
) {
    match stype {
        SemanticType::Categorical => {
            for row_idx in 0..col.len() {
                if let Ok(AnyValue::Null) = col.get(row_idx) {
                    continue;
                }
                if let Some(val_str) = get_string_value(col, row_idx) {
                    let text = format!("{} is {}", col_name, val_str);
                    texts.insert(text);
                }
            }
        }
        SemanticType::Text => {
            if let Ok(str_col) = col.str() {
                for opt_val in str_col.into_iter() {
                    if let Some(s) = opt_val {
                        if !s.is_empty() {
                            texts.insert(s.to_string());
                        }
                    }
                }
            } else {
                for row_idx in 0..col.len() {
                    if let Some(s) = get_string_value(col, row_idx) {
                        if !s.is_empty() {
                            texts.insert(s);
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

// ============================================================================
// Polars Dtype -> SemanticType
// ============================================================================

fn dtype_to_stype(dtype: &DataType) -> Option<SemanticType> {
    match dtype {
        DataType::Boolean => Some(SemanticType::Categorical),
        DataType::Categorical(_, _) | DataType::Enum(_, _) => Some(SemanticType::Categorical),
        DataType::String => Some(SemanticType::Text),
        DataType::Datetime(_, _) | DataType::Date => Some(SemanticType::Timestamp),
        DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Int128
        | DataType::Float32
        | DataType::Float64
        | DataType::Duration(_)
        | DataType::Time => Some(SemanticType::Numerical),
        _ => None,
    }
}

fn determine_stype(col: &Column, col_name: &str, metadata: Option<&TableMetadata>) -> SemanticType {
    if let Some(meta) = metadata {
        if let Some(override_str) = meta.stype_overrides.get(col_name) {
            if let Some(stype) = SemanticType::from_str(override_str) {
                return stype;
            }
        }
    }
    match dtype_to_stype(col.dtype()) {
        Some(stype) => stype,
        None => {
            warn!(
                "Unknown dtype {:?} for column '{}', defaulting to Text",
                col.dtype(),
                col_name
            );
            SemanticType::Text
        }
    }
}

// ============================================================================
// Running Statistics
// ============================================================================

#[derive(Default, Clone, Copy)]
struct RunningStats {
    count: u64,
    mean: f64,
    m2: f64,
}

impl RunningStats {
    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn std(&self) -> f64 {
        if self.count > 1 {
            (self.m2 / (self.count - 1) as f64).sqrt().max(1e-8)
        } else {
            1.0
        }
    }
}

// ============================================================================
// Column Statistics
// ============================================================================

struct ColumnStats {
    name: String,
    dtype: DataType,
    stype: SemanticType,
    mean: Option<f32>,
    std: Option<f32>,
}

// ============================================================================
// Table Info
// ============================================================================

struct TableInfo {
    name: String,
    path: PathBuf,
    num_rows: usize,
    metadata: Option<TableMetadata>,
    column_stats: Vec<ColumnStats>,
}

// ============================================================================
// Utilities
// ============================================================================

fn path_to_str(path: &PathBuf) -> Result<&str> {
    path.to_str()
        .with_context(|| format!("Path contains invalid UTF-8: {:?}", path))
}

fn should_ignore_column(col_name: &str, ignored_cols: &[&str]) -> bool {
    if ignored_cols.contains(&col_name) {
        return true;
    }
    if col_name.starts_with("Unnamed: ") {
        if col_name[9..].parse::<u32>().is_ok() {
            return true;
        }
    }
    false
}

fn iter_datetime_as_epoch_secs<F: FnMut(Option<i64>)>(col: &Column, mut f: F) {
    match col.dtype() {
        DataType::Datetime(unit, _tz) => {
            if let Ok(ca) = col.datetime() {
                for opt_val in ca.phys.into_iter() {
                    let epoch = opt_val.map(|v| match unit {
                        TimeUnit::Nanoseconds => v / 1_000_000_000,
                        TimeUnit::Microseconds => v / 1_000_000,
                        TimeUnit::Milliseconds => v / 1_000,
                    });
                    f(epoch);
                }
            }
        }
        DataType::Date => {
            if let Ok(ca) = col.date() {
                for opt_val in ca.phys.into_iter() {
                    f(opt_val.map(|days| days as i64 * 86400));
                }
            }
        }
        _ => {}
    }
}

fn extract_numerical(col: &Column, mean: f64, std: f64) -> Vec<PackedCell> {
    let n = col.len();
    let mut values = Vec::with_capacity(n);

    if let Ok(f64_col) = col.cast(&DataType::Float64) {
        if let Ok(ca) = f64_col.f64() {
            for opt_val in ca.into_iter() {
                match opt_val {
                    Some(v) if v.is_finite() => {
                        let normalized = ((v - mean) / std.max(1e-8)) as f32;
                        values.push(pack_numerical(normalized));
                    }
                    _ => values.push(pack_null()),
                }
            }
            return values;
        }
    }

    values.resize(n, pack_null());
    values
}

fn extract_categorical_streaming(
    col: &Column,
    col_name: &str,
    vocab: &StreamingVocab,
) -> Vec<PackedCell> {
    let n = col.len();
    let mut values = Vec::with_capacity(n);

    for row_idx in 0..n {
        if let Ok(AnyValue::Null) = col.get(row_idx) {
            values.push(pack_null());
            continue;
        }
        let val_str = get_string_value(col, row_idx).unwrap_or_else(|| "unknown".into());
        let text = format!("{} is {}", col_name, val_str);
        if let Some(idx) = vocab.lookup_text(&text) {
            values.push(pack_embedding_idx(idx));
        } else {
            values.push(pack_null());
        }
    }

    values
}

fn extract_timestamp(col: &Column) -> Vec<PackedCell> {
    let n = col.len();
    let mut values = Vec::with_capacity(n);

    iter_datetime_as_epoch_secs(col, |opt_epoch| match opt_epoch {
        Some(epoch_secs) => values.push(pack_timestamp(epoch_secs as f32)),
        None => values.push(pack_null()),
    });

    if values.is_empty() {
        values.resize(n, pack_null());
    }

    values
}

fn extract_text_streaming(col: &Column, vocab: &StreamingVocab) -> Vec<PackedCell> {
    let n = col.len();
    let mut values = Vec::with_capacity(n);

    if let Ok(str_col) = col.str() {
        for opt_val in str_col.into_iter() {
            match opt_val {
                Some(s) if !s.is_empty() => {
                    if let Some(idx) = vocab.lookup_text(s) {
                        values.push(pack_embedding_idx(idx));
                    } else {
                        values.push(pack_null());
                    }
                }
                _ => values.push(pack_null()),
            }
        }
    } else {
        for row_idx in 0..n {
            if let Ok(AnyValue::Null) = col.get(row_idx) {
                values.push(pack_null());
            } else if let Some(s) = get_string_value(col, row_idx) {
                if s.is_empty() {
                    values.push(pack_null());
                } else if let Some(idx) = vocab.lookup_text(&s) {
                    values.push(pack_embedding_idx(idx));
                } else {
                    values.push(pack_null());
                }
            } else {
                values.push(pack_null());
            }
        }
    }

    values
}

fn extract_column_streaming(
    col: &Column,
    schema_col: &SchemaColumn,
    vocab: &StreamingVocab,
) -> Vec<PackedCell> {
    match schema_col.stype {
        SemanticType::Numerical => {
            let mean = schema_col.norm_mean.unwrap_or(0.0) as f64;
            let std = schema_col.norm_std.unwrap_or(1.0) as f64;
            extract_numerical(col, mean, std)
        }
        SemanticType::Categorical => extract_categorical_streaming(col, &schema_col.name, vocab),
        SemanticType::Timestamp => extract_timestamp(col),
        SemanticType::Text => extract_text_streaming(col, vocab),
    }
}

fn get_string_value(col: &Column, row_idx: usize) -> Option<String> {
    match col.get(row_idx) {
        Ok(AnyValue::Null) => None,
        Ok(AnyValue::String(s)) => Some(s.to_string()),
        Ok(AnyValue::Boolean(b)) => Some(b.to_string()),
        Ok(v) => Some(format!("{}", v)),
        Err(_) => None,
    }
}

fn iter_integer_column<F: FnMut(usize, i64)>(
    col: &Column,
    col_name: &str,
    mut f: F,
) -> (usize, usize) {
    let mut handled = 0usize;
    let mut skipped = 0usize;

    match col.dtype() {
        DataType::Int64 => {
            if let Ok(ca) = col.i64() {
                for (row_idx, opt_val) in ca.into_iter().enumerate() {
                    match opt_val {
                        Some(v) => {
                            f(row_idx, v);
                            handled += 1;
                        }
                        None => skipped += 1,
                    }
                }
            }
        }
        DataType::Int32 => {
            if let Ok(ca) = col.i32() {
                for (row_idx, opt_val) in ca.into_iter().enumerate() {
                    match opt_val {
                        Some(v) => {
                            f(row_idx, v as i64);
                            handled += 1;
                        }
                        None => skipped += 1,
                    }
                }
            }
        }
        DataType::UInt64 => {
            if let Ok(ca) = col.u64() {
                for (row_idx, opt_val) in ca.into_iter().enumerate() {
                    match opt_val {
                        Some(v) => {
                            f(row_idx, v as i64);
                            handled += 1;
                        }
                        None => skipped += 1,
                    }
                }
            }
        }
        DataType::UInt32 => {
            if let Ok(ca) = col.u32() {
                for (row_idx, opt_val) in ca.into_iter().enumerate() {
                    match opt_val {
                        Some(v) => {
                            f(row_idx, v as i64);
                            handled += 1;
                        }
                        None => skipped += 1,
                    }
                }
            }
        }
        DataType::String => {
            if let Ok(ca) = col.str() {
                for (row_idx, opt_val) in ca.into_iter().enumerate() {
                    match opt_val {
                        Some(s) => {
                            if let Ok(v) = s.parse::<i64>() {
                                f(row_idx, v);
                                handled += 1;
                            } else {
                                skipped += 1;
                            }
                        }
                        None => skipped += 1,
                    }
                }
            }
        }
        dtype => {
            warn!(
                "Column '{}' has unsupported dtype {:?} for PK/FK - skipping",
                col_name, dtype
            );
            skipped = col.len();
        }
    }

    (handled, skipped)
}

// ============================================================================
// Phase 1: Schema and Statistics
// ============================================================================

fn get_parquet_row_count(path: &PathBuf) -> Result<usize> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open parquet file: {:?}", path))?;
    let reader = SerializedFileReader::new(file)
        .with_context(|| format!("Failed to read parquet metadata: {:?}", path))?;
    let metadata = reader.metadata();
    let num_rows: i64 = (0..metadata.num_row_groups())
        .map(|i| metadata.row_group(i).num_rows())
        .sum();
    Ok(num_rows as usize)
}

fn collect_schema_and_stats(
    input_dir: &PathBuf,
    metadata: &DatabaseMetadata,
    chunk_size: usize,
) -> Result<(Vec<TableInfo>, RunningStats)> {
    let db_dir = input_dir.join("db");
    let mut tables = Vec::new();
    let mut global_ts_stats = RunningStats::default();

    let entries: Vec<_> = std::fs::read_dir(&db_dir)
        .with_context(|| format!("Failed to read directory: {:?}", db_dir))?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s == "parquet")
                .unwrap_or(false)
        })
        .collect();

    if entries.is_empty() {
        bail!("No parquet files found in {:?}", db_dir);
    }

    let pb = ProgressBar::new(entries.len() as u64);
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} Scanning: {pos}/{len} tables").unwrap(),
    );

    for entry in entries {
        let path = entry.path();
        let path_str = path_to_str(&path)?;
        let table_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let table_metadata = metadata.get(&table_name).cloned();
        let ignored_cols: Vec<&str> = table_metadata
            .as_ref()
            .map(|m| m.ignored_columns.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default();

        let num_rows = get_parquet_row_count(&path)?;

        let schema_df = LazyFrame::scan_parquet(PlPath::new(path_str), ScanArgsParquet::default())?
            .slice(0, 1)
            .collect()
            .with_context(|| format!("Failed to read schema from: {:?}", path))?;

        let mut column_running_stats: HashMap<String, (DataType, SemanticType, RunningStats)> =
            HashMap::new();

        for col in schema_df.get_columns() {
            let col_name = col.name().to_string();
            if should_ignore_column(&col_name, &ignored_cols) {
                continue;
            }
            let stype = determine_stype(col, &col_name, table_metadata.as_ref());
            let dtype = col.dtype().clone();
            column_running_stats.insert(col_name, (dtype, stype, RunningStats::default()));
        }

        let mut offset: i64 = 0;
        let chunk_size_i64 = chunk_size as i64;

        while (offset as usize) < num_rows {
            let df = LazyFrame::scan_parquet(PlPath::new(path_str), ScanArgsParquet::default())?
                .slice(offset, chunk_size_i64 as u32)
                .collect()
                .with_context(|| {
                    format!("Failed to read chunk at offset {} from: {:?}", offset, path)
                })?;

            if df.height() == 0 {
                break;
            }

            for col in df.get_columns() {
                let col_name = col.name().to_string();
                let Some(entry) = column_running_stats.get_mut(&col_name) else {
                    continue;
                };

                if entry.1 == SemanticType::Numerical {
                    if let Ok(f64_col) = col.cast(&DataType::Float64) {
                        if let Ok(ca) = f64_col.f64() {
                            for opt_val in ca.into_iter() {
                                if let Some(val) = opt_val {
                                    if val.is_finite() {
                                        entry.2.update(val);
                                    }
                                }
                            }
                        }
                    }
                }

                if entry.1 == SemanticType::Timestamp {
                    iter_datetime_as_epoch_secs(col, |opt_epoch| {
                        if let Some(epoch_secs) = opt_epoch {
                            global_ts_stats.update(epoch_secs as f64);
                        }
                    });
                }
            }

            offset += df.height() as i64;
        }

        let mut column_stats = Vec::new();
        for col in schema_df.get_columns() {
            let col_name = col.name().to_string();
            if should_ignore_column(&col_name, &ignored_cols) {
                continue;
            }

            if let Some((dtype, stype, stats)) = column_running_stats.remove(&col_name) {
                let (mean, std) = if stype == SemanticType::Numerical {
                    (Some(stats.mean() as f32), Some(stats.std() as f32))
                } else {
                    (None, None)
                };

                column_stats.push(ColumnStats {
                    name: col_name,
                    dtype,
                    stype,
                    mean,
                    std,
                });
            }
        }

        tables.push(TableInfo {
            name: table_name,
            path,
            num_rows,
            metadata: table_metadata,
            column_stats,
        });

        pb.inc(1);
    }

    pb.finish_and_clear();
    tables.sort_by(|a, b| a.name.cmp(&b.name));
    Ok((tables, global_ts_stats))
}

// ============================================================================
// Phase 2: Build Schema
// ============================================================================

fn build_schema(
    table_infos: &[TableInfo],
    embedder: &Embedder,
    global_ts_stats: &RunningStats,
) -> (Schema, HashMap<String, TableIdx>, HashMap<(TableIdx, String), ColumnIdx>) {
    let mut schema = Schema::new();
    let mut table_name_to_idx: HashMap<String, TableIdx> = HashMap::new();
    let mut column_name_to_idx: HashMap<(TableIdx, String), ColumnIdx> = HashMap::new();

    if global_ts_stats.count > 0 {
        schema.timestamp_mean = Some(global_ts_stats.mean());
        schema.timestamp_std = Some(global_ts_stats.std());
    }

    let mut global_col_idx: u32 = 0;
    let mut global_row_idx: u32 = 0;
    let mut column_descriptions: Vec<(ColumnIdx, String)> = Vec::new();

    for (table_idx, info) in table_infos.iter().enumerate() {
        let table_idx = TableIdx(table_idx as u32);
        table_name_to_idx.insert(info.name.clone(), table_idx);

        let col_start = ColumnIdx(global_col_idx);
        let row_start = RowIdx(global_row_idx);

        let pk_col_name = info
            .metadata
            .as_ref()
            .and_then(|m| m.primary_key_column.as_ref());
        let time_col_name = info.metadata.as_ref().and_then(|m| m.time_column.as_ref());
        let mut pk_column_idx: Option<ColumnIdx> = None;
        let mut time_column_idx: Option<ColumnIdx> = None;

        if let Some(pk_name) = pk_col_name {
            if !info.column_stats.iter().any(|c| &c.name == pk_name) {
                warn!(
                    "Table '{}': primary_key_column '{}' not found in schema",
                    info.name, pk_name
                );
            }
        }
        if let Some(time_name) = time_col_name {
            if !info.column_stats.iter().any(|c| &c.name == time_name) {
                warn!(
                    "Table '{}': time_column '{}' not found in schema",
                    info.name, time_name
                );
            }
        }

        for col_stats in &info.column_stats {
            let col_idx = ColumnIdx(global_col_idx);
            column_name_to_idx.insert((table_idx, col_stats.name.clone()), col_idx);

            let is_pk = pk_col_name.map(|s| s == &col_stats.name).unwrap_or(false);
            if is_pk {
                pk_column_idx = Some(col_idx);
            }
            if time_col_name.map(|s| s == &col_stats.name).unwrap_or(false) {
                time_column_idx = Some(col_idx);
            }

            let description = format!("{}.{}", info.name, col_stats.name);
            column_descriptions.push((col_idx, description));

            schema.columns.push(SchemaColumn {
                name: col_stats.name.clone(),
                idx: col_idx,
                table_idx,
                stype: col_stats.stype,
                is_primary_key: is_pk,
                fk_target_column: None,
                norm_mean: col_stats.mean,
                norm_std: col_stats.std,
            });

            global_col_idx += 1;
        }

        let col_end = ColumnIdx(global_col_idx);
        let row_end = RowIdx(global_row_idx + info.num_rows as u32);

        schema.tables.push(Table {
            name: info.name.clone(),
            idx: table_idx,
            column_range: (col_start, col_end),
            feature_columns: Vec::new(),
            row_range: (row_start, row_end),
            primary_key_column: pk_column_idx,
            time_column: time_column_idx,
        });

        global_row_idx += info.num_rows as u32;
    }

    schema.init_column_embeddings(embedder.embedding_dim() as u32);

    if !column_descriptions.is_empty() {
        info!(
            "Embedding {} column descriptions...",
            column_descriptions.len()
        );
        let descriptions: Vec<&str> = column_descriptions
            .iter()
            .map(|(_, s)| s.as_str())
            .collect();
        match embedder.embed_batch_chunked_f16(&descriptions, embedder.config.batch_size) {
            Ok(embeddings) => {
                for ((col_idx, _), embedding) in column_descriptions.iter().zip(embeddings) {
                    schema.set_column_embedding(*col_idx, &embedding);
                }
            }
            Err(e) => {
                warn!("Failed to embed column descriptions: {}", e);
            }
        }
    }

    resolve_foreign_keys(
        table_infos,
        &mut schema,
        &table_name_to_idx,
        &column_name_to_idx,
    );

    compute_feature_columns(&mut schema);

    (schema, table_name_to_idx, column_name_to_idx)
}

fn resolve_foreign_keys(
    table_infos: &[TableInfo],
    schema: &mut Schema,
    table_name_to_idx: &HashMap<String, TableIdx>,
    column_name_to_idx: &HashMap<(TableIdx, String), ColumnIdx>,
) {
    for info in table_infos {
        let Some(ref meta) = info.metadata else {
            continue;
        };
        let table_idx = table_name_to_idx[&info.name];

        for (fk_col_name, target_table_name) in &meta.foreign_key_column_to_primary_key_table {
            let Some(&fk_col_idx) = column_name_to_idx.get(&(table_idx, fk_col_name.clone()))
            else {
                warn!(
                    "Table '{}': FK column '{}' not found in schema",
                    info.name, fk_col_name
                );
                continue;
            };

            let Some(&target_table_idx) = table_name_to_idx.get(target_table_name) else {
                warn!(
                    "Table '{}': FK target table '{}' not found",
                    info.name, target_table_name
                );
                continue;
            };

            let Some(pk_col_idx) = schema.tables[target_table_idx.0 as usize].primary_key_column
            else {
                warn!(
                    "Table '{}': FK target table '{}' has no primary key",
                    info.name, target_table_name
                );
                continue;
            };

            schema.columns[fk_col_idx.0 as usize].fk_target_column = Some(pk_col_idx);
            debug!("FK: {}.{} -> {}", info.name, fk_col_name, target_table_name);
        }
    }
}

fn compute_feature_columns(schema: &mut Schema) {
    for table in schema.tables.iter_mut() {
        let mut feature_cols = Vec::new();
        for col_idx in table.column_range.0 .0..table.column_range.1 .0 {
            let col = &schema.columns[col_idx as usize];
            if !col.is_primary_key && col.fk_target_column.is_none() {
                feature_cols.push(ColumnIdx(col_idx));
            }
        }

        debug!(
            "Table {}: {} feature columns (of {} total)",
            table.name,
            feature_cols.len(),
            table.num_columns()
        );

        table.feature_columns = feature_cols;
    }
}

fn print_schema_summary(table_infos: &[TableInfo]) {
    info!("Schema Summary:");
    for info in table_infos {
        info!("  Table: {} ({} rows)", info.name, info.num_rows);
        for col_stats in &info.column_stats {
            info!(
                "    {:30} | {:20} -> {:?}",
                col_stats.name,
                format!("{:?}", col_stats.dtype),
                col_stats.stype
            );
        }
    }
}

// ============================================================================
// PK Index
// ============================================================================

struct PkIndex {
    index: HashMap<(TableIdx, i64), RowIdx>,
}

impl PkIndex {
    fn new() -> Self {
        Self {
            index: HashMap::new(),
        }
    }

    fn register(&mut self, table_idx: TableIdx, pk_value: i64, row_idx: RowIdx) {
        self.index.insert((table_idx, pk_value), row_idx);
    }

    fn lookup(&self, table_idx: TableIdx, pk_value: i64) -> Option<RowIdx> {
        self.index.get(&(table_idx, pk_value)).copied()
    }

    fn len(&self) -> usize {
        self.index.len()
    }
}

// ============================================================================
// Phase 4: Process Tables
// ============================================================================

fn process_tables_streaming(
    table_infos: &[TableInfo],
    schema: &Schema,
    cells: &mut Cells,
    vocab: &StreamingVocab,
    pk_index: &mut PkIndex,
    chunk_size: usize,
) -> Result<()> {
    let total_rows: usize = table_infos.iter().map(|t| t.num_rows).sum();
    let total_cells: usize = table_infos
        .iter()
        .map(|t| t.num_rows * t.column_stats.len())
        .sum();
    info!(
        "Processing {} tables ({} total rows, {} total cells)...",
        table_infos.len(),
        total_rows,
        total_cells
    );

    cells.reserve(total_cells, total_rows);

    let pb = ProgressBar::new(total_rows as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} rows ({per_sec}, ETA: {eta})",
        )
        .unwrap()
        .progress_chars("█▓▒░  "),
    );

    for (table_idx, info) in table_infos.iter().enumerate() {
        let table_idx = TableIdx(table_idx as u32);
        pb.set_message(format!("{}", info.name));

        let path_str = path_to_str(&info.path)?;
        let row_start = schema.tables[table_idx.0 as usize].row_range.0 .0;

        let ignored_cols: Vec<&str> = info
            .metadata
            .as_ref()
            .map(|m| m.ignored_columns.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default();

        let feature_columns = schema.tables[table_idx.0 as usize].feature_columns.clone();
        let num_feature_cols = feature_columns.len();

        let pk_col_name = schema.tables[table_idx.0 as usize]
            .primary_key_column
            .map(|idx| schema.columns[idx.0 as usize].name.clone());

        let time_col_name = schema.tables[table_idx.0 as usize]
            .time_column
            .map(|idx| schema.columns[idx.0 as usize].name.clone());

        let mut offset: i64 = 0;
        let chunk_size_u32 = chunk_size as u32;
        let mut rows_processed_in_table: u32 = 0;

        while (offset as usize) < info.num_rows {
            let df = LazyFrame::scan_parquet(PlPath::new(path_str), ScanArgsParquet::default())?
                .slice(offset, chunk_size_u32)
                .collect()
                .with_context(|| {
                    format!(
                        "Failed to read chunk at offset {} from: {:?}",
                        offset, info.path
                    )
                })?;

            let batch_rows = df.height();
            if batch_rows == 0 {
                break;
            }

            let polars_columns: HashMap<&str, &Column> = df
                .get_columns()
                .iter()
                .filter(|c| !should_ignore_column(c.name(), &ignored_cols))
                .map(|c| (c.name().as_str(), c))
                .collect();

            let mut column_data: Vec<Vec<PackedCell>> = Vec::with_capacity(num_feature_cols);
            for &col_idx in &feature_columns {
                let schema_col = &schema.columns[col_idx.0 as usize];
                if let Some(polars_col) = polars_columns.get(schema_col.name.as_str()) {
                    let values = extract_column_streaming(polars_col, schema_col, vocab);
                    column_data.push(values);
                } else {
                    warn!("Feature column '{}' not found in parquet", schema_col.name);
                    column_data.push(vec![pack_null(); batch_rows]);
                }
            }

            let row_timestamps =
                extract_row_timestamps_batch(&df, time_col_name.as_deref(), batch_rows);

            let mut row_buffer = Vec::with_capacity(num_feature_cols);
            for row_idx in 0..batch_rows {
                row_buffer.clear();
                for col_data in &column_data {
                    row_buffer.push(col_data[row_idx]);
                }
                cells.push_row(&row_buffer, row_timestamps[row_idx]);
            }

            if let Some(ref pk_name) = pk_col_name {
                if let Some(col) = polars_columns.get(pk_name.as_str()) {
                    let batch_row_start = row_start + rows_processed_in_table;
                    build_pk_index(col, pk_name, table_idx, batch_row_start, pk_index);
                }
            }

            rows_processed_in_table += batch_rows as u32;
            offset += batch_rows as i64;
            pb.set_position((row_start + rows_processed_in_table) as u64);
        }
    }

    pb.finish_with_message("Done");
    Ok(())
}

fn extract_row_timestamps_batch(
    df: &DataFrame,
    time_col_name: Option<&str>,
    num_rows: usize,
) -> Vec<i64> {
    let Some(time_col_name) = time_col_name else {
        return vec![NO_TIMESTAMP; num_rows];
    };

    let Ok(col) = df.column(time_col_name) else {
        warn!("Time column '{}' not found in dataframe", time_col_name);
        return vec![NO_TIMESTAMP; num_rows];
    };

    let mut timestamps = Vec::with_capacity(num_rows);
    iter_datetime_as_epoch_secs(col, |opt_epoch| {
        timestamps.push(opt_epoch.unwrap_or(NO_TIMESTAMP));
    });

    if timestamps.is_empty() {
        warn!(
            "Time column '{}' has unsupported dtype {:?}",
            time_col_name,
            col.dtype()
        );
        timestamps.resize(num_rows, NO_TIMESTAMP);
    }

    timestamps
}

fn build_pk_index(
    col: &Column,
    col_name: &str,
    table_idx: TableIdx,
    row_start: u32,
    pk_index: &mut PkIndex,
) {
    let (handled, skipped) = iter_integer_column(col, col_name, |row_idx, pk| {
        pk_index.register(table_idx, pk, RowIdx(row_start + row_idx as u32));
    });

    if skipped > 0 {
        debug!(
            "PK column '{}': indexed {} values, skipped {} nulls/invalid",
            col_name, handled, skipped
        );
    }
}

// ============================================================================
// Phase 5: Build FK Edges
// ============================================================================

fn build_fk_edges(
    table_infos: &[TableInfo],
    schema: &Schema,
    graph: &mut Graph,
    table_name_to_idx: &HashMap<String, TableIdx>,
    pk_index: &PkIndex,
    chunk_size: usize,
) -> Result<()> {
    let total_fk_cols: usize = table_infos
        .iter()
        .filter_map(|info| info.metadata.as_ref())
        .map(|m| m.foreign_key_column_to_primary_key_table.len())
        .sum();

    if total_fk_cols == 0 {
        info!("No foreign key edges to build");
        let num_rows: usize = table_infos.iter().map(|t| t.num_rows).sum();
        graph.build_from_edges(num_rows, Vec::new());
        return Ok(());
    }

    let pb = ProgressBar::new(total_fk_cols as u64);
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} Building FK edges: {pos}/{len} columns")
            .unwrap(),
    );

    let mut all_edges: Vec<(u32, u32)> = Vec::new();
    let mut total_orphaned = 0usize;

    for (table_idx, info) in table_infos.iter().enumerate() {
        let table_idx = TableIdx(table_idx as u32);
        let Some(ref meta) = info.metadata else {
            continue;
        };
        if meta.foreign_key_column_to_primary_key_table.is_empty() {
            continue;
        }

        let fk_col_names: Vec<String> = meta
            .foreign_key_column_to_primary_key_table
            .keys()
            .cloned()
            .collect();

        let path_str = path_to_str(&info.path)?;
        let row_start = schema.tables[table_idx.0 as usize].row_range.0 .0;

        let mut offset: i64 = 0;
        let chunk_size_u32 = chunk_size as u32;
        let mut rows_processed: u32 = 0;

        while (offset as usize) < info.num_rows {
            let select_cols: Vec<Expr> = fk_col_names.iter().map(|s| col(s.as_str())).collect();

            let df = LazyFrame::scan_parquet(PlPath::new(path_str), ScanArgsParquet::default())?
                .select(select_cols)
                .slice(offset, chunk_size_u32)
                .collect()
                .with_context(|| {
                    format!(
                        "Failed to read FK chunk at offset {} from: {:?}",
                        offset, info.path
                    )
                })?;

            let batch_rows = df.height();
            if batch_rows == 0 {
                break;
            }

            let batch_row_start = row_start + rows_processed;

            for (fk_col_name, target_table_name) in &meta.foreign_key_column_to_primary_key_table {
                let Some(&target_table_idx) = table_name_to_idx.get(target_table_name) else {
                    continue;
                };

                if let Ok(fk_col) = df.column(fk_col_name) {
                    let orphaned = collect_fk_edges(
                        fk_col,
                        fk_col_name,
                        target_table_idx,
                        batch_row_start,
                        pk_index,
                        &mut all_edges,
                    );
                    total_orphaned += orphaned;
                }
            }

            rows_processed += batch_rows as u32;
            offset += batch_rows as i64;
        }

        pb.inc(meta.foreign_key_column_to_primary_key_table.len() as u64);
    }

    pb.finish_and_clear();

    if total_orphaned > 0 {
        warn!(
            "{} FK values had no matching PK (orphaned references)",
            total_orphaned
        );
    }

    info!("Created {} FK edges", all_edges.len());
    let num_rows: usize = table_infos.iter().map(|t| t.num_rows).sum();
    graph.build_from_edges(num_rows, all_edges);

    Ok(())
}

fn collect_fk_edges(
    fk_col: &Column,
    col_name: &str,
    target_table: TableIdx,
    row_start: u32,
    pk_index: &PkIndex,
    edges: &mut Vec<(u32, u32)>,
) -> usize {
    let mut orphaned = 0usize;

    iter_integer_column(fk_col, col_name, |row_idx, fk_val| {
        if let Some(target_row) = pk_index.lookup(target_table, fk_val) {
            edges.push((row_start + row_idx as u32, target_row.0));
        } else {
            orphaned += 1;
        }
    });

    orphaned
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set subscriber");

    let args = Args::parse();

    // Load metadata
    let metadata_path = args.input_dir.join("metadata.json");
    let metadata = if metadata_path.exists() {
        info!("Loading metadata from: {:?}", metadata_path);
        load_metadata(&metadata_path).expect("Failed to load metadata")
    } else {
        info!("No metadata.json found, using defaults");
        HashMap::new()
    };

    // Get database name
    let db_name = args
        .input_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("database");

    // Create output directory for this database
    let output_dir = args.output_dir.join(db_name);
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    // Initialize embedder
    info!("Initializing embedder...");
    let embedder = Embedder::new(EmbedderConfig::default()).expect("Failed to initialize embedder");
    info!("Embedder ready");

    // Phase 1: Schema and stats
    info!(
        "=== Phase 1: Collecting schema and statistics (chunk_size={}) ===",
        args.chunk_size
    );
    let (table_infos, global_ts_stats) =
        collect_schema_and_stats(&args.input_dir, &metadata, args.chunk_size)?;
    info!(
        "Found {} tables, {} total rows",
        table_infos.len(),
        table_infos.iter().map(|t| t.num_rows).sum::<usize>()
    );

    // Phase 2: Build schema
    info!("=== Phase 2: Building schema ===");
    let (schema, table_name_to_idx, _column_name_to_idx) =
        build_schema(&table_infos, &embedder, &global_ts_stats);
    info!(
        "Schema: {} tables, {} columns",
        schema.num_tables(),
        schema.num_columns()
    );
    print_schema_summary(&table_infos);

    // Phase 3: Extract vocabulary (streaming)
    info!("=== Phase 3: Extracting vocabulary (streaming) ===");
    let mut vocab = StreamingVocab::new(&args.output_dir, db_name)?;
    let vocab_size = vocab.extract_vocabulary(&table_infos, &schema, args.chunk_size)?;
    info!("Vocabulary: {} unique texts", vocab_size);

    // Phase 4: Embed vocabulary (streaming to disk)
    info!("=== Phase 4: Embedding vocabulary (streaming to disk) ===");
    vocab.embed_vocabulary(&embedder)?;

    // Phase 5: Process tables (cell encoding)
    info!("=== Phase 5: Processing tables ===");
    let mut cells = Cells::new();
    let mut pk_index = PkIndex::new();
    process_tables_streaming(
        &table_infos,
        &schema,
        &mut cells,
        &vocab,
        &mut pk_index,
        args.chunk_size,
    )?;
    info!(
        "Rows: {}, PK index: {} entries",
        cells.num_rows(),
        pk_index.len()
    );

    // Phase 6: Build FK edges
    info!("=== Phase 6: Building FK edges ===");
    let mut graph = Graph::new();
    build_fk_edges(
        &table_infos,
        &schema,
        &mut graph,
        &table_name_to_idx,
        &pk_index,
        args.chunk_size,
    )?;

    // Cleanup temp files
    vocab.cleanup()?;

    // Phase 7: Save all files
    info!("=== Phase 7: Saving files ===");

    let save_pb = ProgressBar::new_spinner();
    save_pb.set_style(ProgressStyle::with_template("{spinner:.green} {msg}").unwrap());
    save_pb.enable_steady_tick(std::time::Duration::from_millis(100));

    save_pb.set_message("Saving schema.rkyv...");
    schema.save(output_dir.join("schema.rkyv"))?;

    save_pb.set_message("Saving graph.rkyv...");
    graph.save(output_dir.join("graph.rkyv"))?;

    save_pb.set_message("Saving cells.rkyv...");
    cells.save(output_dir.join("cells.rkyv"))?;

    save_pb.set_message("Saving manifest.json...");
    let manifest = Manifest {
        version: "1.0".to_string(),
        created: chrono::Utc::now().to_rfc3339(),
        source_dir: args.input_dir.to_string_lossy().to_string(),
        stats: ManifestStats {
            num_tables: schema.num_tables(),
            num_columns: schema.num_columns(),
            num_rows: cells.num_rows(),
            num_edges: graph.num_edges(),
            vocab_size: vocab.vocab_size,
            embed_dim: vocab.embed_dim,
        },
    };
    manifest.save(output_dir.join("manifest.json"))?;

    save_pb.finish_with_message("Saved!");

    info!("=== Complete ===");
    info!("Output directory: {:?}", output_dir);
    info!(
        "  - schema.rkyv:     {} tables, {} columns",
        schema.num_tables(),
        schema.num_columns()
    );
    info!(
        "  - graph.rkyv:      {} nodes, {} edges",
        graph.num_nodes(),
        graph.num_edges()
    );
    info!("  - cells.rkyv:      {} rows", cells.num_rows());
    info!(
        "  - embeddings.bin:  {} texts, dim={}",
        vocab.vocab_size, vocab.embed_dim
    );
    info!("  - manifest.json:   metadata");

    Ok(())
}
