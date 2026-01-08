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
    Column as SchemaColumn, ColumnIdx, Database, Embedder, EmbedderConfig, ForeignKeyEdge,
    NormalizedCellValue, Row, RowIdx, SemanticType, Table, TableIdx, TableType, TextIdx,
};
use clap::Parser;
use half::f16;
use parquet::file::reader::{FileReader, SerializedFileReader};
use polars::prelude::{Column as PolarsColumn, *};
use tracing::{Level, debug, info, warn};
use tracing_subscriber::FmtSubscriber;

struct TableMeta {
    name: String,
    primary_key: Option<String>, // not sure if this should be optional because we want all tables to have a primary key
    foreign_keys: HashMap<String, String>, // col_name -> target_table_name
    time_column: Option<String>,
}

/// Auto-map polars dtype to our semantic types
/// It is possible to override this mapping in the metadata.json file for each table.
/// For example, if you know that a column is categorical, but the underlying datatype is text, you could override the
/// auto-mapping there (to "categorical").
/// Similarly, if you know that a column is, say, an identifier, but it's stored as a UInt32, you could override the
/// auto-mapping as well (to "unsupported").
fn dtype_to_stype(dtype: &DataType) -> SemanticType {
    match dtype {
        DataType::Boolean => SemanticType::Categorical,
        DataType::Categorical(_, _) => SemanticType::Categorical,
        DataType::Enum(_, _) => SemanticType::Categorical,

        DataType::String => SemanticType::Text,

        DataType::Datetime(_, _) => SemanticType::Datetime,
        DataType::Date => SemanticType::Datetime,

        // TODO(mrdmnd): *should* UInts get automatically turned into Numerical? Or are these likely to be identifiers?
        DataType::UInt8 => SemanticType::Numerical,
        DataType::UInt16 => SemanticType::Numerical,
        DataType::UInt32 => SemanticType::Numerical,
        DataType::UInt64 => SemanticType::Numerical,
        DataType::Int8 => SemanticType::Numerical,
        DataType::Int16 => SemanticType::Numerical,
        DataType::Int32 => SemanticType::Numerical,
        DataType::Int64 => SemanticType::Numerical,
        DataType::Int128 => SemanticType::Numerical,
        DataType::Float32 => SemanticType::Numerical,
        DataType::Float64 => SemanticType::Numerical,
        DataType::Decimal(_, _) => SemanticType::Numerical,
        DataType::Duration(_) => SemanticType::Numerical,

        // Everything else is definitely unsupported... but let's be explicit about it.
        _ => SemanticType::Unsupported,
    }
}

// ============================================================================
// Streaming stats + datetime extraction helpers
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

    fn has_samples(&self) -> bool {
        self.count > 0
    }
}

// From a polars column that is *known to be* a datetime somehow, extract the number of seconds since epoch.
fn extract_datetime_seconds(series: &PolarsColumn, row_idx: usize) -> Option<f32> {
    use polars::prelude::TimeUnit;

    match series.dtype() {
        DataType::Date => series
            .cast(&DataType::Int32)
            .ok()?
            .i32()
            .ok()?
            .get(row_idx)
            .map(|days| days as f32 * 86_400.0),
        DataType::Datetime(time_unit, _) => series
            .cast(&DataType::Int64)
            .ok()?
            .i64()
            .ok()?
            .get(row_idx)
            .map(|raw| match time_unit {
                TimeUnit::Nanoseconds => raw as f32 / 1_000_000_000.0,
                TimeUnit::Microseconds => raw as f32 / 1_000_000.0,
                TimeUnit::Milliseconds => raw as f32 / 1_000.0,
            }),
        _ => series
            .cast(&DataType::Float64)
            .ok()?
            .f64()
            .ok()?
            .get(row_idx)
            .map(|v| v as f32),
    }
}

/// Parse metadata.json to extract relational info from the tables.
fn parse_datbase_metadata(metadata_file: &PathBuf) -> ParquetTableMeta {
    let file = File::open(metadata_file).expect("Failed to open metadata file");
    let metadata = serde_json::from_reader(file).expect("Failed to parse metadata file");

    let mut primary_key = None;
    let mut foreign_keys = HashMap::new();
    let mut time_column = None;

    metadata
}

/// Collect parquet files and metadata from a dataset directory.
/// Expects structure:
///   <data_dir>/tables/*.parquet - core database tables
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

    sources
}

/// Load all parquet files and build the unified Database.
/// Column descriptions and text values are embedded inline using CUDA.
fn load_database(database_dir: &PathBuf, embedder: &Embedder) -> Database {
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
    // Track FK targets per column so we can resolve to target PK column later without storing on Column.
    let mut pending_fk_targets: Vec<Option<TableIdx>> = Vec::new();

    for (table_idx, (parquet_file, table_name, table_type)) in parquet_sources.iter().enumerate() {
        let meta = parse_parquet_metadata(parquet_file, table_name.clone(), *table_type);
        table_key_to_idx.insert((table_name.clone(), *table_type), TableIdx(table_idx));
        db_table_name_to_idx.insert(table_name.clone(), TableIdx(table_idx));

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
    let mut global_col_idx = 0usize;
    let mut global_row_idx = 0usize;

    for (table_idx, (meta, df)) in table_metas.iter().zip(dataframes.iter()).enumerate() {
        let table_idx = TableIdx(table_idx);

        let col_start = ColumnIdx(global_col_idx);
        let row_start = RowIdx(global_row_idx);

        // Build columns for this table
        let mut pk_col: Option<ColumnIdx> = None;
        let mut time_col: Option<ColumnIdx> = None;

        for (_local_idx, field) in df.schema().iter_fields().enumerate() {
            let col_idx = ColumnIdx(global_col_idx);
            let col_name = field.name();

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
            let fk_target_table = meta
                .foreign_keys
                .get(col_name.as_str())
                .and_then(|target_table| db_table_name_to_idx.get(target_table).copied());

            let column = SchemaColumn {
                name: col_name.to_string(),
                idx: col_idx,
                table_idx,
                dtype: dtype_to_stype(field.dtype()),
                is_primary_key: is_pk,
                fk_target_column: None, // Will be resolved later
                column_description_embedding,
                norm_mean: None, // Will be computed during preprocessing
                norm_std: None,  // Will be computed during preprocessing
            };

            db.columns.push(column);
            pending_fk_targets.push(fk_target_table);
            global_col_idx += 1;
        }

        let col_end = ColumnIdx(global_col_idx);
        let num_rows = df.height() as u32;
        let row_end = RowIdx(global_row_idx + num_rows); // careful with the usize + u32 here?

        let table = Table {
            name: meta.name.clone(),
            idx: table_idx,
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
    for (col, fk_target_table) in db.columns.iter_mut().zip(pending_fk_targets.iter()) {
        if let Some(target_table_idx) = fk_target_table {
            let target_table = &db.tables[target_table_idx.0 as usize];
            col.fk_target_column = target_table.primary_key_col;
        }
    }

    // Third pass: gather statistics and build PK index without storing raw cells
    info!("Computing column statistics and primary keys...");
    let mut col_stats: Vec<RunningStats> = vec![RunningStats::default(); db.columns.len()];
    let mut datetime_stats = RunningStats::default();

    for (table_num, df) in dataframes.iter().enumerate() {
        let table_idx = TableIdx(table_num);
        let row_start = db.tables[table_num].row_range.0.0;
        let col_start = db.tables[table_num].column_range.0.0;

        for row_num in 0..df.height() {
            let row_idx = RowIdx(row_start + row_num); // careful with the usize + usize here?

            for col_offset in 0..df.width() {
                let series = df
                    .select_at_idx(col_offset)
                    .expect("column index out of bounds");
                let col_idx = ColumnIdx(col_start + col_offset); // careful with the usize + usize here?
                let column = &db.columns[col_idx.0];

                match column.dtype {
                    SemanticType::Text => {}
                    SemanticType::Number => {
                        let val = series.cast(&DataType::Float64).unwrap();
                        if let Some(v) = val.f64().unwrap().get(row_num) {
                            col_stats[col_idx.0 as usize].update(v);
                            if column.is_primary_key {
                                db.pk_index.insert((table_idx, v as i64), row_idx);
                            }
                        }
                    }
                    SemanticType::Boolean => {
                        if let Some(v) = series.bool().unwrap().get(row_num) {
                            let fv = if v { 1.0 } else { 0.0 };
                            col_stats[col_idx.0 as usize].update(fv);
                        }
                    }
                    SemanticType::Datetime => {
                        if let Some(ts) = extract_datetime_seconds(series, row_num) {
                            datetime_stats.update(ts as f64);
                            db.update_timestamp_range(ts);
                        }
                    }
                }
            }
        }
    }

    // Store normalization parameters
    db.datetime_norm_mean = datetime_stats
        .has_samples()
        .then(|| datetime_stats.mean() as f32);
    db.datetime_norm_std = datetime_stats
        .has_samples()
        .then(|| datetime_stats.std() as f32);

    for (col_idx, col) in db.columns.iter_mut().enumerate() {
        match col.dtype {
            SemanticType::Number | SemanticType::Boolean => {
                col.norm_mean = Some(col_stats[col_idx].mean() as f32);
                col.norm_std = Some(col_stats[col_idx].std() as f32);
            }
            _ => {
                col.norm_mean = None;
                col.norm_std = None;
            }
        }
    }

    // Fourth pass: build rows with normalized values and FK edges (no raw storage)
    info!("Building rows with normalized values and foreign keys...");
    db.rows.reserve(global_row_idx as usize);
    db.fk_edges.clear();
    let mut text_cell_count = 0usize;

    let datetime_mean = db.datetime_norm_mean.unwrap_or(0.0);
    let datetime_std = db.datetime_norm_std.unwrap_or(1.0).max(1e-8);

    for (table_num, df) in dataframes.iter().enumerate() {
        let table_idx = TableIdx(table_num as u32);
        let row_start = db.tables[table_num].row_range.0.0;
        let col_start = db.tables[table_num].column_range.0.0;
        let time_col_local_idx = db.tables[table_num]
            .time_col
            .map(|tc| (tc.0 - col_start) as usize);

        for row_num in 0..df.height() {
            let row_idx = RowIdx(row_start + row_num as u32);

            let mut normalized_cells = Vec::with_capacity(df.width());
            let mut raw_timestamp: Option<f32> = None;

            for col_offset in 0..df.width() {
                let series = df
                    .select_at_idx(col_offset)
                    .expect("column index out of bounds");
                let col_idx = ColumnIdx(col_start + col_offset as u32);
                let column_dtype = db.columns[col_idx.0 as usize].dtype;
                let fk_target_column = db.columns[col_idx.0 as usize].fk_target_column;
                let column_mean = db.columns[col_idx.0 as usize].norm_mean;
                let column_std = db.columns[col_idx.0 as usize].norm_std;
                let mut fk_value: Option<i64> = None;

                let normalized = match column_dtype {
                    SemanticType::Text => match series.str().unwrap().get(row_num) {
                        Some(s) => {
                            text_cell_count += 1;
                            let idx = embed_ctx.intern_text(&mut db, s);
                            NormalizedCellValue::Text(idx)
                        }
                        None => NormalizedCellValue::Null,
                    },
                    SemanticType::Number => {
                        let val = series.cast(&DataType::Float64).unwrap();
                        match val.f64().unwrap().get(row_num) {
                            Some(v) => {
                                fk_value = Some(v as i64);
                                let mean = column_mean.unwrap_or(0.0);
                                let std = column_std.unwrap_or(1.0).max(1e-8);
                                NormalizedCellValue::Scalar(((v as f32) - mean) / std)
                            }
                            None => NormalizedCellValue::Null,
                        }
                    }
                    SemanticType::Datetime => match extract_datetime_seconds(series, row_num) {
                        Some(ts) => {
                            if Some(col_offset) == time_col_local_idx {
                                raw_timestamp = Some(ts);
                            }
                            NormalizedCellValue::Scalar((ts - datetime_mean) / datetime_std)
                        }
                        None => NormalizedCellValue::Null,
                    },
                    SemanticType::Boolean => match series.bool().unwrap().get(row_num) {
                        Some(v) => {
                            let val = if v { 1.0 } else { 0.0 };
                            let mean = column_mean.unwrap_or(0.0);
                            let std = column_std.unwrap_or(1.0).max(1e-8);
                            NormalizedCellValue::Scalar((val - mean) / std)
                        }
                        None => NormalizedCellValue::Null,
                    },
                };

                if let Some(target_column_idx) = fk_target_column {
                    if let Some(fk_num) = fk_value {
                        let target_table_idx = db.columns[target_column_idx.0 as usize].table_idx;
                        if let Some(&target_row_idx) = db.pk_index.get(&(target_table_idx, fk_num))
                        {
                            let edge = ForeignKeyEdge {
                                from_row: row_idx,
                                from_col: col_idx,
                                to_row: target_row_idx,
                            };
                            db.fk_edges.push(edge);
                        }
                    }
                }

                normalized_cells.push(normalized);
            }

            db.rows.push(Row {
                idx: row_idx,
                table_idx,
                raw_timestamp,
                normalized: normalized_cells,
            });
        }
    }

    // Build adjacency lists from collected edges
    db.rebuild_adjacency();

    // Flush any remaining pending text embeddings
    embed_ctx.finalize(&mut db);

    info!(
        "Discovered {} text cells, {} unique text values",
        text_cell_count,
        db.vocab_size()
    );

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
    info!("Initializing embedder...");
    let embedder = Embedder::new(EmbedderConfig::default()).expect("Failed to initialize embedder");
    info!("Embedder initialized successfully");

    // Load database with inline embedding
    info!("Loading database from: {:?}", args.data_dir);
    info!("  DB tables from: {:?}", args.data_dir.join("db"));
    let db = load_database(&args.data_dir, &embedder);

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
            .fk_target_column
            .map(|c| {
                let target_table = db.get_column(c).table_idx;
                format!(" -> {}", db.table_name(target_table))
            })
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
