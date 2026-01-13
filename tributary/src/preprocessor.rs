//! Preprocessor binary: transforms raw parquet databases into graph representation.
//!
//! Optimized for speed and memory efficiency:
//! - Columnar processing using Polars vectorized operations
//! - Pre-allocation of all vectors
//! - Streaming table processing (one table in memory at a time)
//!
//! Usage:
//!   cargo run --release --bin preprocessor -- \
//!     --input-dir databases_raw/rel-event \
//!     --output-dir databases_preprocessed/ \
//!     --verbose

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use tracing::{Level, debug, info, warn};
use tracing_subscriber::FmtSubscriber;

use tributary::{
    Column as SchemaColumn, ColumnIdx, Database, DatabaseMetadata, Embedder, EmbedderConfig,
    EmbeddingIdx, NO_TIMESTAMP, PackedCell, PreprocessingContext, RowIdx, SemanticType, Table,
    TableIdx, TableMetadata, load_metadata, pack_embedding_idx, pack_null, pack_numerical,
    pack_timestamp,
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
    /// Smaller values use less memory but are slower.
    /// Default: 100000 rows per chunk.
    #[arg(long, default_value = "100000")]
    chunk_size: usize,
}

// ============================================================================
// Polars Dtype -> SemanticType
// ============================================================================

/// Maps Polars data types to our semantic types.
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

/// Determines the semantic type for a column, respecting metadata overrides.
fn determine_stype(col: &Column, col_name: &str, metadata: Option<&TableMetadata>) -> SemanticType {
    // Check for metadata override first
    if let Some(meta) = metadata {
        if let Some(override_str) = meta.stype_overrides.get(col_name) {
            if let Some(stype) = SemanticType::from_str(override_str) {
                return stype;
            }
        }
    }
    // Fall back to dtype inference
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
// Running Statistics (Welford's online algorithm)
// ============================================================================

/// Computes mean and std incrementally without storing all values.
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

/// Statistics collected for a single column during schema discovery.
struct ColumnStats {
    name: String,
    dtype: DataType,
    stype: SemanticType,
    mean: Option<f32>,
    std: Option<f32>,
}

// ============================================================================
// Table Info (lightweight metadata for each table)
// ============================================================================

struct TableInfo {
    name: String,
    path: PathBuf,
    num_rows: usize,
    metadata: Option<TableMetadata>,
    column_stats: Vec<ColumnStats>,
}

// ============================================================================
// Path Utilities
// ============================================================================

/// Converts a PathBuf to a string, returning an error if the path is invalid UTF-8.
fn path_to_str(path: &PathBuf) -> Result<&str> {
    path.to_str()
        .with_context(|| format!("Path contains invalid UTF-8: {:?}", path))
}

// ============================================================================
// Column Filtering
// ============================================================================

/// Returns true if the column should be ignored.
/// Ignores:
/// - Columns explicitly listed in metadata's ignored_columns
/// - Pandas index columns (pattern: "Unnamed: N" where N is a number)
fn should_ignore_column(col_name: &str, ignored_cols: &[&str]) -> bool {
    if ignored_cols.contains(&col_name) {
        return true;
    }
    // Skip pandas-style unnamed index columns: "Unnamed: 0", "Unnamed: 1", etc.
    if col_name.starts_with("Unnamed: ") {
        if col_name[9..].parse::<u32>().is_ok() {
            return true;
        }
    }
    false
}

// ============================================================================
// Timestamp Conversion
// ============================================================================

/// Iterates over a datetime column, calling the callback with epoch seconds for each value.
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

// ============================================================================
// Column Data Extraction (per semantic type)
// ============================================================================

/// Extracts numerical column values, normalizing by mean/std.
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

    // Fallback: all nulls
    values.resize(n, pack_null());
    values
}

/// Extracts categorical column values as "column_name is value" text embeddings.
fn extract_categorical(
    col: &Column,
    col_name: &str,
    ctx: &mut PreprocessingContext,
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
        let idx = ctx.intern_text(&text);
        values.push(pack_embedding_idx(idx));
    }

    values
}

/// Extracts timestamp column values as epoch seconds (stored as f32).
fn extract_timestamp(col: &Column) -> Vec<PackedCell> {
    let n = col.len();
    let mut values = Vec::with_capacity(n);

    iter_datetime_as_epoch_secs(col, |opt_epoch| match opt_epoch {
        Some(epoch_secs) => values.push(pack_timestamp(epoch_secs as f32)),
        None => values.push(pack_null()),
    });

    // If iter didn't produce values (unsupported dtype), fill with nulls
    if values.is_empty() {
        values.resize(n, pack_null());
    }

    values
}

/// Extracts text column values as embedding indices.
fn extract_text(col: &Column, ctx: &mut PreprocessingContext) -> Vec<PackedCell> {
    let n = col.len();
    let mut values = Vec::with_capacity(n);

    if let Ok(str_col) = col.str() {
        for opt_val in str_col.into_iter() {
            match opt_val {
                Some(s) if !s.is_empty() => {
                    let idx = ctx.intern_text(s);
                    values.push(pack_embedding_idx(idx));
                }
                _ => values.push(pack_null()),
            }
        }
    } else {
        // Fallback: convert each value to string
        for row_idx in 0..n {
            if let Ok(AnyValue::Null) = col.get(row_idx) {
                values.push(pack_null());
            } else if let Some(s) = get_string_value(col, row_idx) {
                if s.is_empty() {
                    values.push(pack_null());
                } else {
                    let idx = ctx.intern_text(&s);
                    values.push(pack_embedding_idx(idx));
                }
            } else {
                values.push(pack_null());
            }
        }
    }

    values
}

/// Extracts all cell values for a column, dispatching to the appropriate type handler.
fn extract_column(
    col: &Column,
    schema_col: &SchemaColumn,
    ctx: &mut PreprocessingContext,
) -> Vec<PackedCell> {
    match schema_col.stype {
        SemanticType::Numerical => {
            let mean = schema_col.norm_mean.unwrap_or(0.0) as f64;
            let std = schema_col.norm_std.unwrap_or(1.0) as f64;
            extract_numerical(col, mean, std)
        }
        SemanticType::Categorical => extract_categorical(col, &schema_col.name, ctx),
        SemanticType::Timestamp => extract_timestamp(col),
        SemanticType::Text => extract_text(col, ctx),
    }
}

/// Converts a cell value to string representation.
fn get_string_value(col: &Column, row_idx: usize) -> Option<String> {
    match col.get(row_idx) {
        Ok(AnyValue::Null) => None,
        Ok(AnyValue::String(s)) => Some(s.to_string()),
        Ok(AnyValue::Boolean(b)) => Some(b.to_string()),
        Ok(v) => Some(format!("{}", v)),
        Err(_) => None,
    }
}

// ============================================================================
// Integer Column Iteration (for PK/FK handling)
// ============================================================================

/// Iterates over an integer column, calling the callback with (row_idx, value) for non-null values.
/// Handles Int32, Int64, UInt32, UInt64, and String columns.
/// Returns the number of null/unhandled values for diagnostics.
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
            // Try to parse string values as integers
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
// Phase 1: Schema and Statistics (Chunked)
// ============================================================================

/// Gets the total row count from a parquet file without loading data.
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

/// Scans all parquet files to collect schema information and statistics.
/// Uses chunked processing to avoid loading entire files into memory.
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

        // Get row count from parquet metadata (no data loading)
        let num_rows = get_parquet_row_count(&path)?;

        // Get schema from first few rows
        let schema_df = LazyFrame::scan_parquet(PlPath::new(path_str), ScanArgsParquet::default())?
            .slice(0, 1)
            .collect()
            .with_context(|| format!("Failed to read schema from: {:?}", path))?;

        // Initialize running stats per column
        let mut column_running_stats: HashMap<String, (DataType, SemanticType, RunningStats)> =
            HashMap::new();

        // Initialize from schema
        for col in schema_df.get_columns() {
            let col_name = col.name().to_string();
            if should_ignore_column(&col_name, &ignored_cols) {
                continue;
            }
            let stype = determine_stype(col, &col_name, table_metadata.as_ref());
            let dtype = col.dtype().clone();
            column_running_stats.insert(col_name, (dtype, stype, RunningStats::default()));
        }

        // Process file in chunks to collect stats
        let mut offset: i64 = 0;
        let chunk_size_i64 = chunk_size as i64;

        while (offset as usize) < num_rows {
            let df = LazyFrame::scan_parquet(PlPath::new(path_str), ScanArgsParquet::default())?
                .slice(offset, chunk_size_i64 as u32)
                .collect()
                .with_context(|| format!("Failed to read chunk at offset {} from: {:?}", offset, path))?;

            if df.height() == 0 {
                break;
            }

            for col in df.get_columns() {
                let col_name = col.name().to_string();
                let Some(entry) = column_running_stats.get_mut(&col_name) else {
                    continue;
                };

                // Update running stats for numerical columns
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

                // Collect timestamp stats for global normalization
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

        // Convert running stats to ColumnStats (preserve column order from schema)
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

/// Builds the database schema from collected table info.
fn build_schema(
    table_infos: &[TableInfo],
    embedder: &Embedder,
    global_ts_stats: &RunningStats,
) -> (
    Database,
    HashMap<String, TableIdx>,
    HashMap<(TableIdx, String), ColumnIdx>,
) {
    let mut db = Database::new();
    let mut table_name_to_idx: HashMap<String, TableIdx> = HashMap::new();
    let mut column_name_to_idx: HashMap<(TableIdx, String), ColumnIdx> = HashMap::new();

    if global_ts_stats.count > 0 {
        db.timestamp_mean = Some(global_ts_stats.mean());
        db.timestamp_std = Some(global_ts_stats.std());
    }

    let mut global_col_idx: u32 = 0;
    let mut global_row_idx: u32 = 0;
    let mut column_descriptions: Vec<(ColumnIdx, String)> = Vec::new();

    for (table_idx, info) in table_infos.iter().enumerate() {
        let table_idx = TableIdx(table_idx as u32);
        table_name_to_idx.insert(info.name.clone(), table_idx);

        let col_start = ColumnIdx(global_col_idx);
        let row_start = RowIdx(global_row_idx);

        // Find PK and time columns from metadata
        let pk_col_name = info
            .metadata
            .as_ref()
            .and_then(|m| m.primary_key_column.as_ref());
        let time_col_name = info.metadata.as_ref().and_then(|m| m.time_column.as_ref());
        let mut pk_column_idx: Option<ColumnIdx> = None;
        let mut time_column_idx: Option<ColumnIdx> = None;

        // Validate metadata references
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

            db.columns.push(SchemaColumn {
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

        db.tables.push(Table {
            name: info.name.clone(),
            idx: table_idx,
            column_range: (col_start, col_end),
            feature_columns: Vec::new(), // Populated after FK resolution
            row_range: (row_start, row_end),
            primary_key_column: pk_column_idx,
            time_column: time_column_idx,
        });

        global_row_idx += info.num_rows as u32;
    }

    // Initialize column embeddings storage
    db.init_column_embeddings(embedder.embedding_dim() as u32);

    // Batch embed column descriptions
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
                    db.set_column_embedding(*col_idx, &embedding);
                }
            }
            Err(e) => {
                warn!("Failed to embed column descriptions: {}", e);
            }
        }
    }

    // Resolve FK references
    resolve_foreign_keys(
        table_infos,
        &mut db,
        &table_name_to_idx,
        &column_name_to_idx,
    );

    // Compute feature_columns for each table (excludes PK and FK columns)
    // Also computes target_columns from metadata
    compute_feature_columns(&mut db, table_infos);

    (db, table_name_to_idx, column_name_to_idx)
}

/// Resolves foreign key references from metadata.
fn resolve_foreign_keys(
    table_infos: &[TableInfo],
    db: &mut Database,
    table_name_to_idx: &HashMap<String, TableIdx>,
    column_name_to_idx: &HashMap<(TableIdx, String), ColumnIdx>,
) {
    for info in table_infos {
        let Some(ref meta) = info.metadata else {
            continue;
        };
        let table_idx = table_name_to_idx[&info.name];

        for (fk_col_name, target_table_name) in &meta.foreign_key_column_to_primary_key_table {
            // Validate FK column exists
            let Some(&fk_col_idx) = column_name_to_idx.get(&(table_idx, fk_col_name.clone()))
            else {
                warn!(
                    "Table '{}': FK column '{}' not found in schema",
                    info.name, fk_col_name
                );
                continue;
            };

            // Validate target table exists
            let Some(&target_table_idx) = table_name_to_idx.get(target_table_name) else {
                warn!(
                    "Table '{}': FK target table '{}' not found",
                    info.name, target_table_name
                );
                continue;
            };

            // Validate target table has a PK
            let Some(pk_col_idx) = db.tables[target_table_idx.0 as usize].primary_key_column else {
                warn!(
                    "Table '{}': FK target table '{}' has no primary key",
                    info.name, target_table_name
                );
                continue;
            };

            db.columns[fk_col_idx.0 as usize].fk_target_column = Some(pk_col_idx);
            debug!("FK: {}.{} -> {}", info.name, fk_col_name, target_table_name);
        }
    }
}

/// Computes which columns are features (not PK or FK) for each table.
fn compute_feature_columns(db: &mut Database, _table_infos: &[TableInfo]) {
    for table in db.tables.iter_mut() {
        let mut feature_cols = Vec::new();
        for col_idx in table.column_range.0.0..table.column_range.1.0 {
            let col = &db.columns[col_idx as usize];
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

// ============================================================================
// Print Schema Summary
// ============================================================================

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
// Phase 3: Process Tables (Chunked)
// ============================================================================

/// Processes all tables, extracting cell data and building PK indices.
/// Uses chunked processing to avoid loading entire files into memory.
fn process_tables(
    table_infos: &[TableInfo],
    db: &mut Database,
    ctx: &mut PreprocessingContext,
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

    // Pre-allocate cell storage
    db.reserve_cells(total_cells, total_rows);

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
        let row_start = db.tables[table_idx.0 as usize].row_range.0.0;

        let ignored_cols: Vec<&str> = info
            .metadata
            .as_ref()
            .map(|m| m.ignored_columns.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default();

        // Get the feature columns for this table
        let feature_columns = db.tables[table_idx.0 as usize].feature_columns.clone();
        let num_feature_cols = feature_columns.len();

        // Get PK column name if present
        let pk_col_name = db.tables[table_idx.0 as usize]
            .primary_key_column
            .map(|idx| db.columns[idx.0 as usize].name.clone());

        // Get time column name if present
        let time_col_name = db.tables[table_idx.0 as usize]
            .time_column
            .map(|idx| db.columns[idx.0 as usize].name.clone());

        // Process file in chunks
        let mut offset: i64 = 0;
        let chunk_size_u32 = chunk_size as u32;
        let mut rows_processed_in_table: u32 = 0;

        while (offset as usize) < info.num_rows {
            let df = LazyFrame::scan_parquet(PlPath::new(path_str), ScanArgsParquet::default())?
                .slice(offset, chunk_size_u32)
                .collect()
                .with_context(|| format!("Failed to read chunk at offset {} from: {:?}", offset, info.path))?;

            let batch_rows = df.height();
            if batch_rows == 0 {
                break;
            }

            // Build column name -> polars column mapping (filtered)
            let polars_columns: HashMap<&str, &Column> = df
                .get_columns()
                .iter()
                .filter(|c| !should_ignore_column(c.name(), &ignored_cols))
                .map(|c| (c.name().as_str(), c))
                .collect();

            // Extract feature column data as packed cells for this batch
            let mut column_data: Vec<Vec<PackedCell>> = Vec::with_capacity(num_feature_cols);
            for &col_idx in &feature_columns {
                let schema_col = &db.columns[col_idx.0 as usize];
                if let Some(polars_col) = polars_columns.get(schema_col.name.as_str()) {
                    let values = extract_column(polars_col, schema_col, ctx);
                    column_data.push(values);
                } else {
                    warn!("Feature column '{}' not found in parquet", schema_col.name);
                    column_data.push(vec![pack_null(); batch_rows]);
                }
            }

            // Extract raw timestamps for this batch
            let row_timestamps = extract_row_timestamps_batch(
                &df,
                time_col_name.as_deref(),
                batch_rows,
            );

            // Build cells row by row from columnar data
            let mut row_buffer = Vec::with_capacity(num_feature_cols);
            for row_idx in 0..batch_rows {
                row_buffer.clear();
                for col_data in &column_data {
                    row_buffer.push(col_data[row_idx]);
                }
                db.push_row(&row_buffer, row_timestamps[row_idx]);
            }

            // Build PK index for this batch
            if let Some(ref pk_name) = pk_col_name {
                if let Some(col) = polars_columns.get(pk_name.as_str()) {
                    let batch_row_start = row_start + rows_processed_in_table;
                    build_pk_index(col, pk_name, table_idx, batch_row_start, ctx);
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

/// Extracts raw i64 timestamps for each row in a batch from the time_column.
/// Returns NO_TIMESTAMP for rows without a time_column or with null values.
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

    // If iteration didn't produce values (unsupported dtype), fill with NO_TIMESTAMP
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

/// Builds the primary key index for a table.
fn build_pk_index(
    col: &Column,
    col_name: &str,
    table_idx: TableIdx,
    row_start: u32,
    ctx: &mut PreprocessingContext,
) {
    let (handled, skipped) = iter_integer_column(col, col_name, |row_idx, pk| {
        ctx.register_pk(table_idx, pk, RowIdx(row_start + row_idx as u32));
    });

    if skipped > 0 {
        debug!(
            "PK column '{}': indexed {} values, skipped {} nulls/invalid",
            col_name, handled, skipped
        );
    }
}

// ============================================================================
// Phase 4: Build FK Edges (Chunked)
// ============================================================================

/// Builds foreign key edges between tables.
/// Uses chunked processing to avoid loading entire files into memory.
fn build_fk_edges(
    table_infos: &[TableInfo],
    db: &mut Database,
    table_name_to_idx: &HashMap<String, TableIdx>,
    ctx: &PreprocessingContext,
    chunk_size: usize,
) -> Result<()> {
    // Count FK columns for progress
    let total_fk_cols: usize = table_infos
        .iter()
        .filter_map(|info| info.metadata.as_ref())
        .map(|m| m.foreign_key_column_to_primary_key_table.len())
        .sum();

    if total_fk_cols == 0 {
        info!("No foreign key edges to build");
        db.build_csr_from_edges(Vec::new());
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
        let row_start = db.tables[table_idx.0 as usize].row_range.0.0;

        // Process file in chunks, selecting only FK columns
        let mut offset: i64 = 0;
        let chunk_size_u32 = chunk_size as u32;
        let mut rows_processed: u32 = 0;

        while (offset as usize) < info.num_rows {
            // Select only the FK columns to minimize memory usage
            let select_cols: Vec<Expr> = fk_col_names.iter().map(|s| col(s.as_str())).collect();

            let df = LazyFrame::scan_parquet(PlPath::new(path_str), ScanArgsParquet::default())?
                .select(select_cols)
                .slice(offset, chunk_size_u32)
                .collect()
                .with_context(|| format!("Failed to read FK chunk at offset {} from: {:?}", offset, info.path))?;

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
                        ctx,
                        &mut all_edges,
                    );
                    total_orphaned += orphaned;
                }
            }

            rows_processed += batch_rows as u32;
            offset += batch_rows as i64;
        }

        // Increment progress by number of FK columns in this table
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
    db.build_csr_from_edges(all_edges);

    Ok(())
}

/// Collects FK edges for a single column. Returns count of orphaned FK values.
fn collect_fk_edges(
    fk_col: &Column,
    col_name: &str,
    target_table: TableIdx,
    row_start: u32,
    ctx: &PreprocessingContext,
    edges: &mut Vec<(u32, u32)>,
) -> usize {
    let mut orphaned = 0usize;

    iter_integer_column(fk_col, col_name, |row_idx, fk_val| {
        if let Some(target_row) = ctx.lookup_pk(target_table, fk_val) {
            edges.push((row_start + row_idx as u32, target_row.0));
        } else {
            orphaned += 1;
        }
    });

    orphaned
}

// ============================================================================
// Phase 5: Embed Text Values
// ============================================================================

/// Embeds all unique text values collected during processing.
fn embed_text_values(db: &mut Database, ctx: &PreprocessingContext, embedder: &Embedder) {
    let pending_texts = &ctx.pending_texts;
    if pending_texts.is_empty() {
        return;
    }

    db.init_embeddings(embedder.embedding_dim() as u32, pending_texts.len());

    let batch_size = embedder.config.batch_size;
    let total = pending_texts.len();

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} texts ({per_sec}, ETA: {eta})",
        )
        .unwrap()
        .progress_chars("█▓▒░  "),
    );

    for chunk_start in (0..total).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(total);
        let chunk: Vec<&str> = pending_texts[chunk_start..chunk_end]
            .iter()
            .map(|s| s.as_str())
            .collect();

        match embedder.embed_batch_f16(&chunk) {
            Ok(embeddings) => {
                for (i, embedding) in embeddings.into_iter().enumerate() {
                    db.set_embedding(EmbeddingIdx((chunk_start + i) as u32), &embedding);
                }
            }
            Err(e) => {
                warn!("Failed to embed batch at {}: {}", chunk_start, e);
            }
        }

        pb.set_position(chunk_end as u64);
    }

    pb.finish_with_message("Done");
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

    // Initialize embedder
    info!("Initializing embedder...");
    let embedder = Embedder::new(EmbedderConfig::default()).expect("Failed to initialize embedder");
    info!("Embedder ready");

    // Initialize preprocessing context
    let mut ctx = PreprocessingContext::new();

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
    let (mut db, table_name_to_idx, _column_name_to_idx) =
        build_schema(&table_infos, &embedder, &global_ts_stats);
    info!(
        "Schema: {} tables, {} columns",
        db.tables.len(),
        db.columns.len()
    );
    print_schema_summary(&table_infos);

    // Phase 3: Process tables
    info!("=== Phase 3: Processing tables ===");
    process_tables(&table_infos, &mut db, &mut ctx, args.chunk_size)?;
    info!(
        "Rows: {}, PK index: {} entries, Unique texts: {}",
        db.num_rows(),
        ctx.pk_index.len(),
        ctx.vocab_size()
    );

    // Phase 4: Build FK edges
    info!("=== Phase 4: Building FK edges ===");
    build_fk_edges(&table_infos, &mut db, &table_name_to_idx, &ctx, args.chunk_size)?;

    // Phase 5: Embed text values
    info!("=== Phase 5: Embedding text values ===");
    embed_text_values(&mut db, &ctx, &embedder);

    // Drop preprocessing context (no longer needed)
    drop(ctx);

    info!(
        "Database complete: {} tables, {} columns, {} rows, {} edges, {} value embeddings",
        db.num_tables(),
        db.num_columns(),
        db.num_rows(),
        db.num_edges(),
        db.vocab_size()
    );

    // Save with progress indicator
    std::fs::create_dir_all(&args.output_dir).expect("Failed to create output directory");
    let db_name = args
        .input_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("database");
    let output_path = args.output_dir.join(format!("{db_name}.rkyv"));

    info!("Saving to: {:?}", output_path);
    let save_pb = ProgressBar::new_spinner();
    save_pb.set_style(ProgressStyle::with_template("{spinner:.green} Saving database...").unwrap());
    save_pb.enable_steady_tick(std::time::Duration::from_millis(100));

    db.save(&output_path).expect("Failed to save database");

    save_pb.finish_with_message("Saved!");
    info!("Done!");

    Ok(())
}
