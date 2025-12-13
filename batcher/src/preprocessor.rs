// Preprocessor binary: loads parquet files into a unified Database structure.
//
// This builds an in-memory graph representation where:
// - Nodes: each row in each table
// - Edges: foreign key relationships between rows
//
// Usage: cargo run --bin preprocessor -- --data-dir /path/to/data/directory

use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

use batcher::{
    Column, ColumnIdx, Database, Embedder, EmbedderConfig, ForeignKeyEdge, RawCellValue, Row,
    RowIdx, SemanticType, Table, TableIdx, TableType, TextIdx,
};
use clap::Parser;
use half::f16;
use parquet::file::reader::{FileReader, SerializedFileReader};
use polars::prelude::*;
use tracing::{Level, debug, info, warn};
use tracing_subscriber::FmtSubscriber;

/// Intermediate structure for parsing parquet metadata
struct ParquetTableMeta {
    name: String,
    table_type: TableType,
    primary_key: Option<String>,
    foreign_keys: HashMap<String, String>, // col_name -> target_table_name
    time_column: Option<String>,
}

/// Map polars dtype to our semantic type
fn dtype_to_semantic(dtype: &DataType) -> SemanticType {
    match dtype {
        DataType::Boolean => SemanticType::Boolean,
        DataType::String => SemanticType::Text,
        DataType::Datetime(_, _) | DataType::Date => SemanticType::Datetime,
        _ => SemanticType::Number, // Int*, UInt*, Float* all become Number
    }
}

// ============================================================================
// Embedding Context - manages streaming text embedding during loading
// ============================================================================

/// Context for managing embeddings during database loading.
/// Accumulates text values and embeds them in batches for efficiency.
struct EmbeddingContext<'a> {
    embedder: &'a Embedder,
    /// Pending texts to embed: (text, TextIdx)
    pending_texts: Vec<(String, TextIdx)>,
    /// Batch size for embedding
    batch_size: usize,
}

impl<'a> EmbeddingContext<'a> {
    fn new(embedder: &'a Embedder) -> Self {
        let batch_size = embedder.config.batch_size;
        Self {
            embedder,
            pending_texts: Vec::with_capacity(batch_size),
            batch_size,
        }
    }

    /// Intern a text value, queueing it for batch embedding.
    fn intern_text(&mut self, db: &mut Database, text: &str) -> TextIdx {
        if let Some(&idx) = db.text_value_lookup.get(text) {
            return idx;
        }

        let idx = TextIdx(db.text_value_lookup.len() as u32);
        db.text_value_lookup.insert(text.to_string(), idx);
        // Push empty placeholder - will be filled when batch is flushed
        db.text_value_embeddings.push(Vec::new());

        self.pending_texts.push((text.to_string(), idx));

        // Flush batch if full
        if self.pending_texts.len() >= self.batch_size {
            self.flush_text_batch(db);
        }

        idx
    }

    /// Flush any pending text embeddings
    fn flush_text_batch(&mut self, db: &mut Database) {
        if self.pending_texts.is_empty() {
            return;
        }

        let texts: Vec<&str> = self.pending_texts.iter().map(|(s, _)| s.as_str()).collect();
        let count = texts.len();

        match self.embedder.embed_batch_f16(&texts) {
            Ok(embeddings) => {
                for ((_, idx), embedding) in self.pending_texts.drain(..).zip(embeddings) {
                    if (idx.0 as usize) < db.text_value_embeddings.len() {
                        db.text_value_embeddings[idx.0 as usize] = embedding;
                    }
                }
                debug!("Embedded {} text values", count);
            }
            Err(e) => {
                warn!("Failed to embed text batch: {}", e);
                self.pending_texts.clear();
            }
        }
    }

    /// Embed a column description synchronously
    fn embed_column_description(&self, table_name: &str, col_name: &str) -> Vec<f16> {
        let description = format!("{} of {}", col_name, table_name);
        self.embedder
            .embed_one_f16(&description)
            .expect("Failed to embed column description")
    }

    /// Finalize: flush any remaining pending texts
    fn finalize(&mut self, db: &mut Database) {
        self.flush_text_batch(db);
    }
}

/// Parse parquet file metadata to extract relational info
fn parse_parquet_metadata(
    parquet_file: &PathBuf,
    table_name: String,
    table_type: TableType,
) -> ParquetTableMeta {
    let file = File::open(parquet_file).expect("Failed to open parquet file");
    let reader = SerializedFileReader::new(file).expect("Failed to create parquet reader");
    let metadata = reader.metadata();
    let file_metadata = metadata.file_metadata();

    let mut primary_key = None;
    let mut foreign_keys = HashMap::new();
    let mut time_column = None;

    if let Some(kv_metadata) = file_metadata.key_value_metadata() {
        for kv in kv_metadata {
            match kv.key.as_str() {
                "pkey_col" => {
                    if let Some(value) = &kv.value {
                        primary_key = serde_json::from_str(value).ok();
                    }
                }
                "fkey_col_to_pkey_table" => {
                    if let Some(value) = &kv.value {
                        if let Ok(fks) = serde_json::from_str::<HashMap<String, String>>(value) {
                            foreign_keys = fks;
                        }
                    }
                }
                "time_col" => {
                    if let Some(value) = &kv.value {
                        time_column = serde_json::from_str(value).ok();
                    }
                }
                _ => {}
            }
        }
    }

    ParquetTableMeta {
        name: table_name,
        table_type,
        primary_key,
        foreign_keys,
        time_column,
    }
}

/// Collect parquet files and metadata from a dataset directory.
/// Expects structure:
///   <data_dir>/db/*.parquet - core database tables
///   <data_dir>/tasks/<task-name>/{train,val,test}.parquet - task tables (optional)
/// Returns (parquet_path, table_name, table_type) for each table.
fn collect_parquet_sources(data_dir: &PathBuf) -> Vec<(PathBuf, String, TableType)> {
    let mut sources = Vec::new();

    // Collect DB tables from data_dir/db/
    let db_dir = data_dir.join("db");
    let mut db_files: Vec<_> = glob::glob(db_dir.join("*.parquet").to_str().unwrap())
        .unwrap()
        .filter_map(|p| p.ok())
        .collect();
    db_files.sort();

    for parquet_file in db_files {
        let name = parquet_file
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        sources.push((parquet_file, name, TableType::Db));
    }

    // Collect task tables from data_dir/tasks/ if it exists
    let tasks_dir = data_dir.join("tasks");
    if tasks_dir.exists() && tasks_dir.is_dir() {
        // Find all task subdirectories
        let mut task_dirs: Vec<_> = std::fs::read_dir(&tasks_dir)
            .expect("Failed to read tasks directory")
            .filter_map(|entry| {
                let entry = entry.ok()?;
                if entry.file_type().ok()?.is_dir() {
                    Some(entry.path())
                } else {
                    None
                }
            })
            .collect();
        task_dirs.sort();

        for task_dir in task_dirs {
            let task_name = task_dir.file_name().unwrap().to_str().unwrap().to_string();

            // Look for train.parquet, val.parquet, test.parquet
            for (split_name, table_type) in [
                ("train", TableType::Train),
                ("val", TableType::Val),
                ("test", TableType::Test),
            ] {
                let split_path = task_dir.join(format!("{}.parquet", split_name));
                if split_path.exists() {
                    // Use task_name as the table name (like relational-transformer)
                    sources.push((split_path, task_name.clone(), table_type));
                }
            }
        }
    }

    sources
}

/// Load all parquet files and build the unified Database.
/// Column descriptions and text values are embedded inline using CUDA.
fn load_database(data_dir: &PathBuf, embedder: &Embedder) -> Database {
    let mut db = Database::new();
    let mut embed_ctx = EmbeddingContext::new(embedder);

    // Collect all parquet sources (db tables + task tables)
    let parquet_sources = collect_parquet_sources(data_dir);

    // First pass: parse metadata and build table/column schema
    let mut table_metas: Vec<ParquetTableMeta> = Vec::new();
    let mut dataframes: Vec<DataFrame> = Vec::new();
    // Map from (table_name, table_type) to TableIdx for FK resolution
    let mut table_key_to_idx: HashMap<(String, TableType), TableIdx> = HashMap::new();
    // Also keep a name-only map for FK lookups (FKs point to Db tables by name)
    let mut db_table_name_to_idx: HashMap<String, TableIdx> = HashMap::new();

    for (table_idx, (parquet_file, table_name, table_type)) in parquet_sources.iter().enumerate() {
        let meta = parse_parquet_metadata(parquet_file, table_name.clone(), *table_type);
        table_key_to_idx.insert(
            (table_name.clone(), *table_type),
            TableIdx(table_idx as u32),
        );

        // For Db tables, also add to the name-only lookup (for FK resolution)
        if *table_type == TableType::Db {
            db_table_name_to_idx.insert(table_name.clone(), TableIdx(table_idx as u32));
        }

        // Load the dataframe
        let file = File::open(parquet_file).expect("Failed to open parquet file");
        let df = ParquetReader::new(file)
            .finish()
            .expect("Failed to read parquet file");

        table_metas.push(meta);
        dataframes.push(df);
    }

    info!(
        "Loaded {} parquet files ({} db tables, {} task table splits), building schema...",
        parquet_sources.len(),
        parquet_sources
            .iter()
            .filter(|(_, _, t)| *t == TableType::Db)
            .count(),
        parquet_sources
            .iter()
            .filter(|(_, _, t)| *t != TableType::Db)
            .count(),
    );

    // Second pass: build tables and columns, embed column descriptions
    let mut global_col_idx = 0u32;
    let mut global_row_idx = 0u32;

    for (table_idx, (meta, df)) in table_metas.iter().zip(dataframes.iter()).enumerate() {
        let table_idx = TableIdx(table_idx as u32);

        let col_start = ColumnIdx(global_col_idx);
        let row_start = RowIdx(global_row_idx);

        // Build columns for this table
        let mut pk_col: Option<ColumnIdx> = None;
        let mut time_col: Option<ColumnIdx> = None;

        for (local_idx, field) in df.schema().iter_fields().enumerate() {
            let col_idx = ColumnIdx(global_col_idx);
            let col_name = field.name();

            // Create schema string for embedding
            let schema_str = format!("{} of {}", col_name, meta.name);
            let schema_idx = embed_ctx.intern_text(&mut db, &schema_str);

            // Embed column description inline
            let column_description_embedding =
                Some(embed_ctx.embed_column_description(&meta.name, col_name));

            // Check if this is PK or time column
            let is_pk = meta.primary_key.as_ref() == Some(&col_name.to_string());
            if is_pk {
                pk_col = Some(col_idx);
            }
            if meta.time_column.as_ref() == Some(&col_name.to_string()) {
                time_col = Some(col_idx);
            }

            // Check if this is a FK - FKs always point to Db tables
            let fk_target = meta
                .foreign_keys
                .get(col_name.as_str())
                .and_then(|target_table| db_table_name_to_idx.get(target_table).copied());

            let column = Column {
                name: col_name.to_string(),
                idx: col_idx,
                local_idx: local_idx as u32,
                table_idx,
                schema_idx,
                dtype: dtype_to_semantic(field.dtype()),
                is_primary_key: is_pk,
                fk_target_table: fk_target,
                fk_target_column: None, // Will be resolved later
                column_description_embedding,
                norm_mean: None, // Will be computed by db.normalize()
                norm_std: None,  // Will be computed by db.normalize()
            };

            db.columns.push(column);
            global_col_idx += 1;
        }

        let col_end = ColumnIdx(global_col_idx);
        let num_rows = df.height() as u32;
        let row_end = RowIdx(global_row_idx + num_rows);

        let table = Table {
            name: meta.name.clone(),
            idx: table_idx,
            table_type: meta.table_type,
            column_range: (col_start, col_end),
            row_range: (row_start, row_end),
            primary_key_col: pk_col,
            time_col,
        };

        db.tables.push(table);
        global_row_idx += num_rows;
    }

    info!(
        "Schema complete: {} columns, {} rows expected",
        global_col_idx, global_row_idx
    );

    // Resolve FK target columns (they reference the PK of the target table)
    for col in &mut db.columns {
        if let Some(target_table_idx) = col.fk_target_table {
            let target_table = &db.tables[target_table_idx.0 as usize];
            col.fk_target_column = target_table.primary_key_col;
        }
    }

    // Third pass: build rows and raw cell values, discover text strings, populate pk_index
    info!("Processing rows and discovering text values...");
    let mut text_columns_count = 0usize;

    for (table_num, df) in dataframes.iter().enumerate() {
        let table_idx = TableIdx(table_num as u32);
        // Copy values out to avoid borrow issues
        let row_start = db.tables[table_num].row_range.0.0;
        let col_start = db.tables[table_num].column_range.0.0;
        let table_type = db.tables[table_num].table_type;
        let is_task_row = table_type != TableType::Db;

        for row_num in 0..df.height() {
            let row_idx = RowIdx(row_start + row_num as u32);

            let mut raw_cells = Vec::with_capacity(df.width());

            for (col_offset, series) in df.get_columns().iter().enumerate() {
                let col_idx = ColumnIdx(col_start + col_offset as u32);
                // Copy column info we need
                let dtype = db.columns[col_idx.0 as usize].dtype;
                let is_pk = db.columns[col_idx.0 as usize].is_primary_key;

                let cell = match dtype {
                    SemanticType::Text => {
                        // Discover text string and embed it, store raw string
                        match series.str().unwrap().get(row_num) {
                            Some(s) => {
                                text_columns_count += 1;
                                // Intern for embedding, but store the raw string
                                embed_ctx.intern_text(&mut db, s);
                                RawCellValue::Text(s.to_string())
                            }
                            None => RawCellValue::Null,
                        }
                    }
                    SemanticType::Number => {
                        let val = series.cast(&DataType::Float64).unwrap();
                        match val.f64().unwrap().get(row_num) {
                            Some(v) => RawCellValue::Number(v as f32),
                            None => RawCellValue::Null,
                        }
                    }
                    SemanticType::Datetime => {
                        // Handle both Date (days since epoch) and Datetime (various TimeUnits)
                        use polars::prelude::TimeUnit;
                        let ts_seconds: Option<f64> = match series.dtype() {
                            DataType::Date => {
                                // Date is stored as i32 days since epoch
                                let casted = series.cast(&DataType::Int32).unwrap();
                                casted
                                    .i32()
                                    .unwrap()
                                    .get(row_num)
                                    .map(|days| days as f64 * 86400.0) // days -> seconds
                            }
                            DataType::Datetime(time_unit, _) => {
                                // Datetime is stored as i64 in the specified TimeUnit
                                let casted = series.cast(&DataType::Int64).unwrap();
                                casted
                                    .i64()
                                    .unwrap()
                                    .get(row_num)
                                    .map(|raw| match time_unit {
                                        TimeUnit::Nanoseconds => raw as f64 / 1_000_000_000.0,
                                        TimeUnit::Microseconds => raw as f64 / 1_000_000.0,
                                        TimeUnit::Milliseconds => raw as f64 / 1_000.0,
                                    })
                            }
                            _ => {
                                // Fallback: treat as already in reasonable units
                                let casted = series.cast(&DataType::Float64).unwrap();
                                casted.f64().unwrap().get(row_num)
                            }
                        };
                        match ts_seconds {
                            Some(ts) => {
                                let ts = ts as f32;
                                db.update_timestamp_range(ts);
                                RawCellValue::Datetime(ts)
                            }
                            None => RawCellValue::Null,
                        }
                    }
                    SemanticType::Boolean => match series.bool().unwrap().get(row_num) {
                        Some(v) => RawCellValue::Boolean(v),
                        None => RawCellValue::Null,
                    },
                };

                // Build PK index
                if is_pk {
                    if let RawCellValue::Number(v) = cell {
                        db.pk_index.insert((table_idx, v as i64), row_idx);
                    }
                }

                raw_cells.push(cell);
            }

            db.rows.push(Row {
                idx: row_idx,
                table_idx,
                is_task_row,
                raw: raw_cells,
                normalized: Vec::new(), // Will be populated by db.normalize()
            });
        }
    }

    // Flush any remaining pending text embeddings
    embed_ctx.finalize(&mut db);

    info!(
        "Discovered {} text cells, {} unique text values",
        text_columns_count,
        db.vocab_size()
    );

    // Initialize edge adjacency lists
    db.edges_from = vec![Vec::new(); db.rows.len()];
    db.edges_to = vec![Vec::new(); db.rows.len()];

    // Fourth pass: build FK edges
    for row in &db.rows {
        let table = &db.tables[row.table_idx.0 as usize];

        for (local_col, cell) in row.raw.iter().enumerate() {
            let col_idx = ColumnIdx(table.column_range.0.0 + local_col as u32);
            let column = &db.columns[col_idx.0 as usize];

            if let Some(target_table_idx) = column.fk_target_table {
                // This is a FK column - look up the target row
                if let RawCellValue::Number(fk_value) = cell {
                    let key = (target_table_idx, *fk_value as i64);
                    if let Some(&target_row_idx) = db.pk_index.get(&key) {
                        let edge = ForeignKeyEdge {
                            from_row: row.idx,
                            from_col: col_idx,
                            to_row: target_row_idx,
                        };
                        let edge_idx = db.fk_edges.len();
                        db.fk_edges.push(edge);

                        db.edges_from[row.idx.0 as usize].push(edge_idx);
                        db.edges_to[target_row_idx.0 as usize].push(edge_idx);
                    }
                }
            }
        }
    }

    db
}

#[derive(Parser, Debug)]
#[command(name = "preprocessor")]
#[command(
    about = "Preprocess parquet data files and metadata into an in-memory graph representation."
)]
struct Args {
    /// Path to dataset directory containing db/ and optionally tasks/ subdirectories.
    /// Structure:
    ///   <data-dir>/db/*.parquet          - core database tables
    ///   <data-dir>/tasks/<task>/{train,val,test}.parquet - task tables (optional)
    #[arg(short, long)]
    data_dir: PathBuf,

    /// Enable verbose debug dump of entire database structure
    #[arg(short, long, default_value = "false")]
    verbose: bool,
}

fn main() {
    // Initialize tracing
    let _subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(true)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .init();

    let args = Args::parse();

    // Initialize CUDA embedder
    info!("Initializing CUDA embedder...");
    let embedder =
        Embedder::new(EmbedderConfig::default()).expect("Failed to initialize CUDA embedder");
    info!("Embedder initialized successfully");

    // Load database with inline embedding
    info!("Loading database from: {:?}", args.data_dir);
    info!("  DB tables from: {:?}", args.data_dir.join("db"));
    let tasks_dir = args.data_dir.join("tasks");
    if tasks_dir.exists() {
        info!("  Task tables from: {:?}", tasks_dir);
    }
    let mut db = load_database(&args.data_dir, &embedder);

    // Normalize all cell values (z-score for numbers/booleans/datetimes, TextIdx for text)
    info!("Normalizing cell values...");
    db.normalize();
    info!("Normalization complete");

    // Save database to .rkyv file in current working directory
    let output_name = args
        .data_dir
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("database");
    let output_path = PathBuf::from(format!("{}.rkyv", output_name));
    info!("Saving database to: {:?}", output_path);
    db.save(&output_path).expect("Failed to save database");
    info!("Database saved successfully");

    // Print summary
    info!(
        "Database: {} tables, {} columns, {} rows, {} vocab, {} FK edges",
        db.num_tables(),
        db.num_columns(),
        db.num_rows(),
        db.vocab_size(),
        db.fk_edges.len()
    );
    if let (Some(min_ts), Some(max_ts)) = (db.min_timestamp, db.max_timestamp) {
        // Convert epoch seconds to human-readable dates
        use chrono::{DateTime, Utc};
        let min_dt = DateTime::<Utc>::from_timestamp(min_ts as i64, 0)
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| format!("{:.0}s", min_ts));
        let max_dt = DateTime::<Utc>::from_timestamp(max_ts as i64, 0)
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| format!("{:.0}s", max_ts));
        info!("Timestamp range: {} to {}", min_dt, max_dt);
    }

    // Print tables
    info!("Tables:");
    for table in &db.tables {
        let pk_display = table
            .primary_key_col
            .map(|c| format!("'{}'", db.column_name(c)))
            .unwrap_or_else(|| "(no pk col)".to_string());
        let time_display = table
            .time_col
            .map(|c| format!("'{}'", db.column_name(c)))
            .unwrap_or_else(|| "(no time col)".to_string());
        let type_display = match table.table_type {
            TableType::Db => "Db",
            TableType::Train => "Train",
            TableType::Val => "Val",
            TableType::Test => "Test",
        };

        info!(
            "  [{}] {} ({}) : {} cols, {} rows, pk_col={}, time_col={}",
            table.idx.0,
            db.table_name(table.idx),
            type_display,
            table.num_columns(),
            table.num_rows(),
            pk_display,
            time_display
        );
    }

    // Print columns
    info!("Columns:");
    for col in &db.columns {
        let fk_info = col
            .fk_target_table
            .map(|t| format!(" -> {}", db.table_name(t)))
            .unwrap_or_default();

        info!(
            "  [{}] {}.{} : {:?}{}{}",
            col.idx.0,
            db.table_name(col.table_idx),
            db.column_name(col.idx),
            col.dtype,
            if col.is_primary_key { " [PK]" } else { "" },
            fk_info
        );
    }

    // Sample vocab entries with embedding status
    info!("Vocab sample (first 15):");
    let mut vocab_items: Vec<_> = db.text_value_lookup.iter().collect();
    vocab_items.sort_by_key(|(_, idx)| idx.0);
    for (text, idx) in vocab_items.iter().take(15) {
        let has_embedding = !db.text_value_embeddings[idx.0 as usize].is_empty();
        let embed_marker = if has_embedding { " ✓" } else { "" };
        info!("  [{}] {:?}{}", idx.0, text, embed_marker);
    }

    // Sample edges
    info!("FK edges sample (first 10):");
    for edge in db.fk_edges.iter().take(10) {
        let from_table = db.get_row(edge.from_row).table_idx;
        let to_table = db.get_row(edge.to_row).table_idx;
        info!(
            "  {}[{}].{} -> {}[{}]",
            db.table_name(from_table),
            edge.from_row.0,
            db.column_name(edge.from_col),
            db.table_name(to_table),
            edge.to_row.0
        );
    }

    // Print embedding stats
    let embedded_text_count = db
        .text_value_embeddings
        .iter()
        .filter(|e| !e.is_empty())
        .count();
    info!(
        "Embedding stats: {}/{} text values, {}/{} column descriptions",
        embedded_text_count,
        db.vocab_size(),
        db.columns
            .iter()
            .filter(|c| c.column_description_embedding.is_some())
            .count(),
        db.columns.len()
    );

    // Verbose dump if requested
    if args.verbose {
        db.dump_verbose();
    }

    info!("Preprocessing complete!");
}
