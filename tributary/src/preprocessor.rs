//! Preprocessor binary: transforms raw parquet databases into graph representation.
//!
//! Loads parquet files from a database directory, applies normalization and embedding,
//! then serializes the result as a .rkyv file for fast loading during training.
//!
//! Usage:
//!   cargo run --release --bin preprocessor -- \
//!     --input-dir databases_raw/rel-event \
//!     --output-dir databases_preprocessed/ \
//!     --verbose

use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

use clap::Parser;
use half::f16;
use polars::prelude::*;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

use tributary::{
    CellValue, Column as SchemaColumn, ColumnIdx, Database, DatabaseMetadata, Embedder,
    EmbedderConfig, EmbeddingIdx, ForeignKeyEdge, Row, RowIdx, SemanticType, Table, TableIdx,
    load_metadata,
};

// ============================================================================
// Running Statistics (for z-score normalization)
// ============================================================================

/// Online mean/variance computation using Welford's algorithm.
#[derive(Debug, Clone, Default)]
struct RunningStats {
    count: usize,
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
        if self.count < 2 {
            1.0
        } else {
            (self.m2 / (self.count - 1) as f64).sqrt().max(1e-8)
        }
    }

    fn has_samples(&self) -> bool {
        self.count > 0
    }
}

// ============================================================================
// Polars Dtype -> SemanticType
// ============================================================================

/// Map Polars dtype to SemanticType (can be overridden via metadata.json).
fn dtype_to_stype(dtype: &DataType) -> SemanticType {
    match dtype {
        // Categorical types
        DataType::Boolean => SemanticType::Categorical,
        DataType::Categorical(_, _) | DataType::Enum(_, _) => SemanticType::Categorical,

        // Text
        DataType::String => SemanticType::Text,

        // Timestamps
        DataType::Datetime(_, _) | DataType::Date => SemanticType::Timestamp,

        // Numeric types
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
        | DataType::Decimal(_, _)
        | DataType::Duration(_) => SemanticType::Numerical,

        // Treat Time as timestamp
        DataType::Time => SemanticType::Timestamp,

        // Default fallback for unsupported types
        _ => SemanticType::Categorical,
    }
}

// ============================================================================
// Datetime Extraction
// ============================================================================

/// Extract epoch seconds from a datetime/date column.
fn extract_epoch_seconds(series: &Column, row_idx: usize) -> Option<f64> {
    match series.dtype() {
        DataType::Date => series
            .cast(&DataType::Int32)
            .ok()?
            .i32()
            .ok()?
            .get(row_idx)
            .map(|days| days as f64 * 86_400.0),

        DataType::Datetime(time_unit, _) => series
            .cast(&DataType::Int64)
            .ok()?
            .i64()
            .ok()?
            .get(row_idx)
            .map(|raw| match time_unit {
                TimeUnit::Nanoseconds => raw as f64 / 1_000_000_000.0,
                TimeUnit::Microseconds => raw as f64 / 1_000_000.0,
                TimeUnit::Milliseconds => raw as f64 / 1_000.0,
            }),

        DataType::Time => series
            .cast(&DataType::Int64)
            .ok()?
            .i64()
            .ok()?
            .get(row_idx)
            .map(|nanos| nanos as f64 / 1_000_000_000.0),

        _ => None,
    }
}

// ============================================================================
// Embedding Context
// ============================================================================

/// Manages batch embedding during database loading.
struct EmbeddingContext<'a> {
    embedder: &'a Embedder,
    pending: Vec<(String, EmbeddingIdx)>,
    batch_size: usize,
}

impl<'a> EmbeddingContext<'a> {
    fn new(embedder: &'a Embedder) -> Self {
        Self {
            embedder,
            pending: Vec::with_capacity(embedder.config.batch_size),
            batch_size: embedder.config.batch_size,
        }
    }

    /// Intern a text value for embedding, batching for efficiency.
    fn intern(&mut self, db: &mut Database, text: &str) -> EmbeddingIdx {
        let idx = db.reserve_text(text);

        // Only queue if this is a new text
        if db.text_embeddings[idx.0].is_empty() {
            self.pending.push((text.to_string(), idx));
            if self.pending.len() >= self.batch_size {
                self.flush(db);
            }
        }

        idx
    }

    /// Flush any pending embeddings.
    fn flush(&mut self, db: &mut Database) {
        if self.pending.is_empty() {
            return;
        }

        let texts: Vec<&str> = self.pending.iter().map(|(s, _)| s.as_str()).collect();

        match self.embedder.embed_batch_f16(&texts) {
            Ok(embeddings) => {
                for ((_, idx), embedding) in self.pending.drain(..).zip(embeddings) {
                    db.set_text_embedding(idx, embedding);
                }
            }
            Err(e) => {
                warn!("Failed to embed batch: {e}");
                self.pending.clear();
            }
        }
    }

    /// Embed a column description directly.
    fn embed_column_description(&self, table_name: &str, col_name: &str) -> Vec<f16> {
        let text = format!("{col_name} of {table_name}");
        self.embedder
            .embed_one_f16(&text)
            .unwrap_or_else(|_| vec![f16::ZERO; self.embedder.embedding_dim()])
    }
}

// ============================================================================
// Parquet Loading
// ============================================================================

/// Collected info about a table from parquet + metadata.
struct TableInfo {
    name: String,
    dataframe: DataFrame,
    primary_key_col: Option<String>,
    foreign_keys: HashMap<String, String>, // col -> target_table
    time_col: Option<String>,
    stype_overrides: HashMap<String, SemanticType>,
    ignored_columns: Vec<String>,
}

/// Collect parquet files and metadata for all tables.
fn collect_tables(
    input_dir: &PathBuf,
    metadata: &DatabaseMetadata,
) -> std::io::Result<Vec<TableInfo>> {
    let db_dir = input_dir.join("db");
    let mut parquet_files: Vec<_> = glob::glob(db_dir.join("*.parquet").to_str().unwrap())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
        .filter_map(|p| p.ok())
        .collect();
    parquet_files.sort();

    let mut tables = Vec::with_capacity(parquet_files.len());

    for path in parquet_files {
        let table_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid filename")
            })?
            .to_string();

        // Load dataframe
        let file = File::open(&path)?;
        let df = ParquetReader::new(file)
            .finish()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        // Get metadata for this table (or use defaults)
        let table_meta = metadata.get(&table_name);

        let primary_key_col = table_meta.and_then(|m| m.primary_key_column.clone());
        let foreign_keys = table_meta
            .map(|m| m.foreign_key_column_to_primary_key_table.clone())
            .unwrap_or_default();
        let time_col = table_meta.and_then(|m| m.time_column.clone());
        let stype_overrides: HashMap<String, SemanticType> = table_meta
            .map(|m| {
                m.stype_overrides
                    .iter()
                    .filter_map(|(k, v)| SemanticType::from_str(v).map(|st| (k.clone(), st)))
                    .collect()
            })
            .unwrap_or_default();
        let ignored_columns = table_meta
            .map(|m| m.ignored_columns.clone())
            .unwrap_or_default();

        tables.push(TableInfo {
            name: table_name,
            dataframe: df,
            primary_key_col,
            foreign_keys,
            time_col,
            stype_overrides,
            ignored_columns,
        });
    }

    Ok(tables)
}

// ============================================================================
// Database Building
// ============================================================================

/// Build the database from collected table info.
fn build_database(tables: Vec<TableInfo>, embedder: &Embedder, verbose: bool) -> Database {
    let mut db = Database::new();
    let mut embed_ctx = EmbeddingContext::new(embedder);

    // Maps for FK resolution
    let mut table_name_to_idx: HashMap<String, TableIdx> = HashMap::new();
    let mut pending_fk_targets: Vec<Option<String>> = Vec::new(); // col_idx -> target_table_name

    let mut global_col_idx = 0usize;
    let mut global_row_idx = 0usize;

    // ========================================================================
    // Pass 1: Build schema (tables + columns)
    // ========================================================================
    info!("Building schema...");

    for (table_num, table_info) in tables.iter().enumerate() {
        let table_idx = TableIdx(table_num);
        table_name_to_idx.insert(table_info.name.clone(), table_idx);

        let col_start = ColumnIdx(global_col_idx);
        let row_start = RowIdx(global_row_idx);

        let mut pk_col_idx: Option<ColumnIdx> = None;
        let mut time_col_idx: Option<ColumnIdx> = None;

        for field in table_info.dataframe.schema().iter_fields() {
            let col_name = field.name().to_string();

            // Skip ignored columns
            if table_info.ignored_columns.contains(&col_name) {
                continue;
            }

            let col_idx = ColumnIdx(global_col_idx);

            // Determine semantic type (with override support)
            let stype = table_info
                .stype_overrides
                .get(&col_name)
                .copied()
                .unwrap_or_else(|| dtype_to_stype(field.dtype()));

            // Check for special columns
            let is_pk = table_info.primary_key_col.as_ref() == Some(&col_name);
            if is_pk {
                pk_col_idx = Some(col_idx);
            }
            if table_info.time_col.as_ref() == Some(&col_name) {
                time_col_idx = Some(col_idx);
            }

            // Track FK target table (to resolve after all tables are indexed)
            let fk_target = table_info.foreign_keys.get(&col_name).cloned();
            pending_fk_targets.push(fk_target);

            // Embed column description
            let description_embedding =
                embed_ctx.embed_column_description(&table_info.name, &col_name);

            let column = SchemaColumn {
                name: col_name,
                idx: col_idx,
                table_idx,
                stype,
                is_primary_key: is_pk,
                fk_target_column: None, // Will be resolved later
                description_embedding,
                norm_mean: None,
                norm_std: None,
                category_values: None,
            };

            db.columns.push(column);
            global_col_idx += 1;
        }

        let col_end = ColumnIdx(global_col_idx);
        let num_rows = table_info.dataframe.height();
        let row_end = RowIdx(global_row_idx + num_rows);

        let table = Table {
            name: table_info.name.clone(),
            idx: table_idx,
            column_range: (col_start, col_end),
            row_range: (row_start, row_end),
            primary_key_column: pk_col_idx,
            time_column: time_col_idx,
        };

        db.tables.push(table);
        global_row_idx += num_rows;
    }

    info!(
        "Schema: {} tables, {} columns, {} rows expected",
        db.tables.len(),
        db.columns.len(),
        global_row_idx
    );

    // ========================================================================
    // Resolve FK target columns
    // ========================================================================
    for (col_idx, fk_target_table) in pending_fk_targets.iter().enumerate() {
        if let Some(target_table_name) = fk_target_table {
            if let Some(&target_table_idx) = table_name_to_idx.get(target_table_name) {
                let target_pk_col = db.tables[target_table_idx.0].primary_key_column;
                db.columns[col_idx].fk_target_column = target_pk_col;
            } else {
                warn!(
                    "FK target table '{}' not found for column '{}'",
                    target_table_name, db.columns[col_idx].name
                );
            }
        }
    }

    // ========================================================================
    // Pass 2: Compute statistics for normalization
    // ========================================================================
    info!("Computing column statistics...");

    let mut col_stats: Vec<RunningStats> = vec![RunningStats::default(); db.columns.len()];
    let mut timestamp_stats = RunningStats::default();

    for table_info in &tables {
        let table_idx = *table_name_to_idx.get(&table_info.name).unwrap();
        let col_start = db.tables[table_idx.0].column_range.0 .0;

        let mut col_offset = 0;
        for field in table_info.dataframe.schema().iter_fields() {
            let col_name = field.name().to_string();
            if table_info.ignored_columns.contains(&col_name) {
                continue;
            }

            let col_idx = col_start + col_offset;
            let stype = db.columns[col_idx].stype;

            let series = table_info
                .dataframe
                .column(&col_name)
                .expect("Column not found");

            match stype {
                SemanticType::Numerical => {
                    if let Ok(vals) = series.cast(&DataType::Float64) {
                        if let Ok(arr) = vals.f64() {
                            for val in arr.into_iter().flatten() {
                                col_stats[col_idx].update(val);
                            }
                        }
                    }
                }
                SemanticType::Timestamp => {
                    for row in 0..series.len() {
                        if let Some(epoch) = extract_epoch_seconds(series, row) {
                            timestamp_stats.update(epoch);
                        }
                    }
                }
                SemanticType::Categorical => {
                    // Collect unique values
                    let unique_vals: Vec<String> = series
                        .unique()
                        .map(|u| {
                            u.iter()
                                .filter_map(|v| v.to_string().parse().ok())
                                .collect()
                        })
                        .unwrap_or_default();
                    db.columns[col_idx].category_values = Some(unique_vals);
                }
                _ => {}
            }

            col_offset += 1;
        }
    }

    // Store normalization parameters
    for (col_idx, stats) in col_stats.iter().enumerate() {
        if stats.has_samples() && db.columns[col_idx].stype == SemanticType::Numerical {
            db.columns[col_idx].norm_mean = Some(stats.mean() as f32);
            db.columns[col_idx].norm_std = Some(stats.std() as f32);
        }
    }

    if timestamp_stats.has_samples() {
        db.timestamp_mean = Some(timestamp_stats.mean());
        db.timestamp_std = Some(timestamp_stats.std());
    }

    // ========================================================================
    // Pass 3: Build rows and FK edges
    // ========================================================================
    info!("Building rows and foreign key edges...");

    db.rows.reserve(global_row_idx);

    let ts_mean = db.timestamp_mean.unwrap_or(0.0);
    let ts_std = db.timestamp_std.unwrap_or(1.0);

    for table_info in &tables {
        let table_idx = *table_name_to_idx.get(&table_info.name).unwrap();
        let row_start = db.tables[table_idx.0].row_range.0 .0;
        let col_start = db.tables[table_idx.0].column_range.0 .0;

        for row_num in 0..table_info.dataframe.height() {
            let row_idx = RowIdx(row_start + row_num);
            let mut values = Vec::new();

            let mut col_offset = 0;
            for field in table_info.dataframe.schema().iter_fields() {
                let col_name = field.name().to_string();
                if table_info.ignored_columns.contains(&col_name) {
                    continue;
                }

                let col_idx = ColumnIdx(col_start + col_offset);
                let column = &db.columns[col_idx.0];
                let series = table_info
                    .dataframe
                    .column(&col_name)
                    .expect("Column not found");

                // Extract and normalize the cell value
                let cell_value = extract_cell_value(
                    series,
                    row_num,
                    column,
                    &mut embed_ctx,
                    &mut db,
                    ts_mean,
                    ts_std,
                );

                // Handle FK edge creation
                if let Some(target_col_idx) = column.fk_target_column {
                    if let CellValue::Numerical(normalized_val) = &cell_value {
                        // Denormalize to get original PK value
                        let mean = column.norm_mean.unwrap_or(0.0);
                        let std = column.norm_std.unwrap_or(1.0);
                        let pk_val = (*normalized_val * std + mean).round() as i64;

                        let target_table_idx = db.columns[target_col_idx.0].table_idx;
                        if let Some(&target_row_idx) = db.pk_index.get(&(target_table_idx, pk_val))
                        {
                            db.fk_edges.push(ForeignKeyEdge {
                                from_row: row_idx,
                                from_col: col_idx,
                                to_row: target_row_idx,
                            });
                        }
                    }
                }

                values.push(cell_value);
                col_offset += 1;
            }

            // Index primary key
            if let Some(pk_col_idx) = db.tables[table_idx.0].primary_key_column {
                let local_pk_idx = pk_col_idx.0 - col_start;
                if let CellValue::Numerical(normalized_val) = &values[local_pk_idx] {
                    let col = &db.columns[pk_col_idx.0];
                    let mean = col.norm_mean.unwrap_or(0.0);
                    let std = col.norm_std.unwrap_or(1.0);
                    let pk_val = (*normalized_val * std + mean).round() as i64;
                    db.pk_index.insert((table_idx, pk_val), row_idx);
                }
            }

            db.rows.push(Row {
                idx: row_idx,
                table_idx,
                values,
            });
        }
    }

    // Flush remaining embeddings
    embed_ctx.flush(&mut db);

    // Build adjacency lists
    db.build_adjacency();

    if verbose {
        info!("Database summary:");
        info!("  Tables: {}", db.num_tables());
        info!("  Columns: {}", db.num_columns());
        info!("  Rows: {}", db.num_rows());
        info!("  FK edges: {}", db.num_edges());
        info!("  Vocab size: {}", db.vocab_size());
    }

    db
}

/// Extract a cell value from a series at a given row.
fn extract_cell_value(
    series: &polars::prelude::Column,
    row: usize,
    column: &SchemaColumn,
    embed_ctx: &mut EmbeddingContext,
    db: &mut Database,
    ts_mean: f64,
    ts_std: f64,
) -> CellValue {
    match column.stype {
        SemanticType::Numerical => {
            if let Ok(vals) = series.cast(&DataType::Float64) {
                if let Ok(arr) = vals.f64() {
                    if let Some(val) = arr.get(row) {
                        let mean = column.norm_mean.unwrap_or(0.0) as f64;
                        let std = column.norm_std.unwrap_or(1.0).max(1e-8) as f64;
                        return CellValue::Numerical(((val - mean) / std) as f32);
                    }
                }
            }
            CellValue::Null
        }

        SemanticType::Categorical => {
            let text = series.get(row).ok().map(|v| v.to_string());
            match text {
                Some(s) if !s.is_empty() && s != "null" => {
                    let formatted = format!("{} is {}", column.name, s);
                    let idx = embed_ctx.intern(db, &formatted);
                    CellValue::Categorical(idx)
                }
                _ => CellValue::Null,
            }
        }

        SemanticType::Timestamp => {
            if let Some(epoch) = extract_epoch_seconds(series, row) {
                CellValue::from_epoch_seconds(epoch, ts_mean, ts_std)
            } else {
                CellValue::Null
            }
        }

        SemanticType::Text => {
            if let Ok(str_arr) = series.str() {
                if let Some(s) = str_arr.get(row) {
                    if !s.is_empty() {
                        let idx = embed_ctx.intern(db, s);
                        return CellValue::Text(idx);
                    }
                }
            }
            CellValue::Null
        }
    }
}

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "preprocessor")]
#[command(about = "Preprocess a database from parquet files into graph representation.")]
struct Args {
    /// Path to the database directory containing db/*.parquet and metadata.json.
    #[arg(short, long)]
    input_dir: PathBuf,

    /// Output directory for the .rkyv file.
    #[arg(short, long, default_value = ".")]
    output_dir: PathBuf,

    /// Enable verbose output.
    #[arg(short, long, default_value = "false")]
    verbose: bool,
}

fn main() {
    // Initialize logging
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

    // Collect tables
    info!("Loading parquet files from: {:?}", args.input_dir.join("db"));
    let tables = collect_tables(&args.input_dir, &metadata).expect("Failed to collect tables");
    info!("Found {} tables", tables.len());

    // Build database
    let db = build_database(tables, &embedder, args.verbose);

    // Save
    let db_name = args
        .input_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("database");
    let output_path = args.output_dir.join(format!("{db_name}.rkyv"));

    info!("Saving to: {:?}", output_path);
    db.save(&output_path).expect("Failed to save database");
    info!("Done!");
}
