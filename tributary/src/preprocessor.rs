//! Preprocessor binary: transforms raw parquet databases into graph representation.
//!
//! Optimized for speed and memory efficiency:
//! - Columnar processing using Polars vectorized operations
//! - Pre-allocation of all vectors
//! - Parallel row building with rayon
//! - Streaming table processing (one table in memory at a time)
//!
//! Usage:
//!   cargo run --release --bin preprocessor -- \
//!     --input-dir databases_raw/rel-event \
//!     --output-dir databases_preprocessed/ \
//!     --verbose

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use tracing::{Level, debug, info, warn};
use tracing_subscriber::FmtSubscriber;

use tributary::{
    Column as SchemaColumn, ColumnIdx, Database, DatabaseMetadata, Embedder, EmbedderConfig,
    EmbeddingIdx, PackedCell, PreprocessingContext, RowIdx, SemanticType, Table, TableIdx,
    TableMetadata, load_metadata, pack_embedding_idx, pack_null, pack_numerical, pack_timestamp,
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
// Running Statistics (Welford's online algorithm)
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
// Table Info (lightweight metadata)
// ============================================================================

struct TableInfo {
    name: String,
    path: PathBuf,
    num_rows: usize,
    metadata: Option<TableMetadata>,
    /// (col_name, dtype, stype, mean, std)
    column_stats: Vec<(String, DataType, SemanticType, Option<f32>, Option<f32>)>,
}

// ============================================================================
// Column Data Extraction (vectorized)
// ============================================================================

/// Extract all cell values for a column using vectorized Polars operations.
/// Returns packed cells (u32) for memory efficiency.
fn extract_column_vectorized(
    col: &Column,
    schema_col: &SchemaColumn,
    ctx: &mut PreprocessingContext,
) -> Vec<PackedCell> {
    let n = col.len();
    let mut values = Vec::with_capacity(n);

    match schema_col.stype {
        SemanticType::Numerical => {
            let mean = schema_col.norm_mean.unwrap_or(0.0) as f64;
            let std = schema_col.norm_std.unwrap_or(1.0) as f64;

            // Try to cast to f64 for vectorized processing
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
            // Fallback
            for _ in 0..n {
                values.push(pack_null());
            }
        }

        SemanticType::Categorical => {
            // Format as "column_name is value" to give embedding model context
            let col_name = &schema_col.name;
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
        }

        SemanticType::Timestamp => {
            match col.dtype() {
                DataType::Datetime(_, _) => {
                    if let Ok(ca) = col.datetime() {
                        for opt_val in ca.phys.into_iter() {
                            match opt_val {
                                Some(v) => {
                                    let epoch_secs = (v as f64 / 1_000_000.0) as f32;
                                    values.push(pack_timestamp(epoch_secs));
                                }
                                None => values.push(pack_null()),
                            }
                        }
                        return values;
                    }
                }
                DataType::Date => {
                    if let Ok(ca) = col.date() {
                        for opt_val in ca.phys.into_iter() {
                            match opt_val {
                                Some(v) => {
                                    let epoch_secs = (v as f64 * 86400.0) as f32;
                                    values.push(pack_timestamp(epoch_secs));
                                }
                                None => values.push(pack_null()),
                            }
                        }
                        return values;
                    }
                }
                DataType::Int64 | DataType::UInt64 => {
                    if let Ok(ca) = col.i64() {
                        for opt_val in ca.into_iter() {
                            match opt_val {
                                Some(v) => {
                                    let epoch_secs = if v > 1_000_000_000_000 {
                                        (v as f64 / 1000.0) as f32
                                    } else {
                                        v as f32
                                    };
                                    values.push(pack_timestamp(epoch_secs));
                                }
                                None => values.push(pack_null()),
                            }
                        }
                        return values;
                    }
                }
                _ => {}
            }
            // Fallback
            for _ in 0..n {
                values.push(pack_null());
            }
        }

        SemanticType::Text => {
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
                // Fallback for non-string columns
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
        }
    }

    values
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

// ============================================================================
// Phase 1: Schema and Statistics
// ============================================================================

fn collect_schema_and_stats(
    input_dir: &PathBuf,
    metadata: &DatabaseMetadata,
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

    let pb = ProgressBar::new(entries.len() as u64);
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} Scanning: {pos}/{len} tables").unwrap(),
    );

    for entry in entries {
        let path = entry.path();
        let table_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let df = LazyFrame::scan_parquet(
            PlPath::new(path.to_str().unwrap()),
            ScanArgsParquet::default(),
        )?
        .collect()
        .with_context(|| format!("Failed to read parquet: {:?}", path))?;

        let num_rows = df.height();
        let table_metadata = metadata.get(&table_name).cloned();
        let ignored_cols: Vec<&str> = table_metadata
            .as_ref()
            .map(|m| m.ignored_columns.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default();

        let mut column_stats = Vec::new();

        for col in df.get_columns() {
            let col_name = col.name().to_string();
            if ignored_cols.contains(&col_name.as_str()) {
                continue;
            }
            // Skip "Unnamed" columns (imputed primary keys from pandas/CSV imports)
            if col_name.starts_with("Unnamed") {
                continue;
            }

            let stype = determine_stype(col, &col_name, table_metadata.as_ref());

            let dtype = col.dtype().clone();

            let (mean, std) = if stype == SemanticType::Numerical {
                let mut stats = RunningStats::default();
                if let Ok(f64_col) = col.cast(&DataType::Float64) {
                    if let Ok(ca) = f64_col.f64() {
                        for opt_val in ca.into_iter() {
                            if let Some(val) = opt_val {
                                if val.is_finite() {
                                    stats.update(val);
                                }
                            }
                        }
                    }
                }
                (Some(stats.mean() as f32), Some(stats.std() as f32))
            } else {
                (None, None)
            };

            if stype == SemanticType::Timestamp {
                collect_timestamp_stats(col, &mut global_ts_stats);
            }

            column_stats.push((col_name, dtype, stype, mean, std));
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

fn collect_timestamp_stats(col: &Column, stats: &mut RunningStats) {
    match col.dtype() {
        DataType::Datetime(_, _) => {
            if let Ok(ca) = col.datetime() {
                for opt_val in ca.phys.into_iter() {
                    if let Some(val) = opt_val {
                        stats.update(val as f64 / 1_000_000.0);
                    }
                }
            }
        }
        DataType::Date => {
            if let Ok(ca) = col.date() {
                for opt_val in ca.phys.into_iter() {
                    if let Some(val) = opt_val {
                        stats.update(val as f64 * 86400.0);
                    }
                }
            }
        }
        DataType::Int64 | DataType::UInt64 => {
            if let Ok(ca) = col.i64() {
                for opt_val in ca.into_iter() {
                    if let Some(val) = opt_val {
                        let epoch_secs = if val > 1_000_000_000_000 {
                            val as f64 / 1000.0
                        } else {
                            val as f64
                        };
                        stats.update(epoch_secs);
                    }
                }
            }
        }
        _ => {}
    }
}

// ============================================================================
// Phase 2: Build Schema
// ============================================================================

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

        let pk_col_name = info
            .metadata
            .as_ref()
            .and_then(|m| m.primary_key_column.as_ref());
        let time_col_name = info.metadata.as_ref().and_then(|m| m.time_column.as_ref());
        let mut pk_column_idx: Option<ColumnIdx> = None;
        let mut time_column_idx: Option<ColumnIdx> = None;

        for (col_name, _dtype, stype, mean, std) in &info.column_stats {
            let col_idx = ColumnIdx(global_col_idx);
            column_name_to_idx.insert((table_idx, col_name.clone()), col_idx);

            let is_pk = pk_col_name.map(|s| s == col_name).unwrap_or(false);
            if is_pk {
                pk_column_idx = Some(col_idx);
            }
            if time_col_name.map(|s| s == col_name).unwrap_or(false) {
                time_column_idx = Some(col_idx);
            }

            let description = format!("{}.{}", info.name, col_name);
            column_descriptions.push((col_idx, description));

            db.columns.push(SchemaColumn {
                name: col_name.clone(),
                idx: col_idx,
                table_idx,
                stype: *stype,
                is_primary_key: is_pk,
                fk_target_column: None,
                norm_mean: *mean,
                norm_std: *std,
            });

            global_col_idx += 1;
        }

        let col_end = ColumnIdx(global_col_idx);
        let row_end = RowIdx(global_row_idx + info.num_rows as u32);

        db.tables.push(Table {
            name: info.name.clone(),
            idx: table_idx,
            column_range: (col_start, col_end),
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
                // Leave as zeros
            }
        }
    }

    // Resolve FK references
    for info in table_infos {
        if let Some(ref meta) = info.metadata {
            let table_idx = table_name_to_idx[&info.name];
            for (fk_col_name, target_table_name) in &meta.foreign_key_column_to_primary_key_table {
                if let Some(&fk_col_idx) = column_name_to_idx.get(&(table_idx, fk_col_name.clone()))
                {
                    if let Some(&target_table_idx) = table_name_to_idx.get(target_table_name) {
                        if let Some(pk_col_idx) =
                            db.tables[target_table_idx.0 as usize].primary_key_column
                        {
                            db.columns[fk_col_idx.0 as usize].fk_target_column = Some(pk_col_idx);
                            debug!("FK: {}.{} -> {}", info.name, fk_col_name, target_table_name);
                        }
                    }
                }
            }
        }
    }

    (db, table_name_to_idx, column_name_to_idx)
}

// ============================================================================
// Print Schema Summary
// ============================================================================

fn print_schema_summary(table_infos: &[TableInfo]) {
    info!("Schema Summary:");
    for info in table_infos {
        info!("  Table: {} ({} rows)", info.name, info.num_rows);
        for (col_name, dtype, stype, _mean, _std) in &info.column_stats {
            info!(
                "    {:30} | {:20} -> {:?}",
                col_name,
                format!("{:?}", dtype),
                stype
            );
        }
    }
}

// ============================================================================
// Phase 3: Process Tables (vectorized, parallel row building)
// ============================================================================

fn process_tables(
    table_infos: &[TableInfo],
    db: &mut Database,
    column_name_to_idx: &HashMap<(TableIdx, String), ColumnIdx>,
    ctx: &mut PreprocessingContext,
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

        let df = LazyFrame::scan_parquet(
            PlPath::new(info.path.to_str().unwrap()),
            ScanArgsParquet::default(),
        )?
        .collect()
        .with_context(|| format!("Failed to read parquet: {:?}", info.path))?;

        let row_start = db.tables[table_idx.0 as usize].row_range.0.0;

        let ignored_cols: Vec<&str> = info
            .metadata
            .as_ref()
            .map(|m| m.ignored_columns.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default();

        let table_columns: Vec<_> = df
            .get_columns()
            .iter()
            .filter(|c| !ignored_cols.contains(&c.name().as_str()))
            .filter(|c| !c.name().starts_with("Unnamed"))
            .collect();

        // Get schema info for each column
        let schema_cols: Vec<SchemaColumn> = table_columns
            .iter()
            .map(|col| {
                let col_name = col.name().to_string();
                let schema_col_idx = column_name_to_idx[&(table_idx, col_name)];
                db.columns[schema_col_idx.0 as usize].clone()
            })
            .collect();

        // Extract all column data vectorized (as packed cells)
        let mut column_data: Vec<Vec<PackedCell>> = Vec::with_capacity(table_columns.len());
        for (col, schema_col) in table_columns.iter().zip(schema_cols.iter()) {
            let values = extract_column_vectorized(col, schema_col, ctx);
            column_data.push(values);
        }

        // Build cells row by row from columnar data
        let num_cols = column_data.len();
        let mut row_buffer = Vec::with_capacity(num_cols);
        for row_idx in 0..info.num_rows {
            row_buffer.clear();
            for col_data in &column_data {
                row_buffer.push(col_data[row_idx]);
            }
            db.push_row(&row_buffer);
        }

        // Build PK index
        if let Some(pk_col_idx) = db.tables[table_idx.0 as usize].primary_key_column {
            let pk_col_name = &db.columns[pk_col_idx.0 as usize].name;
            if let Ok(col) = df.column(pk_col_name) {
                build_pk_index_vectorized(col, table_idx, row_start, ctx);
            }
        }

        pb.set_position((row_start + info.num_rows as u32) as u64);
    }

    pb.finish_with_message("Done");

    Ok(())
}

fn build_pk_index_vectorized(
    col: &Column,
    table_idx: TableIdx,
    row_start: u32,
    ctx: &mut PreprocessingContext,
) {
    match col.dtype() {
        DataType::Int64 => {
            if let Ok(ca) = col.i64() {
                for (row_idx, opt_val) in ca.into_iter().enumerate() {
                    if let Some(pk) = opt_val {
                        ctx.register_pk(table_idx, pk, RowIdx(row_start + row_idx as u32));
                    }
                }
            }
        }
        DataType::Int32 => {
            if let Ok(ca) = col.i32() {
                for (row_idx, opt_val) in ca.into_iter().enumerate() {
                    if let Some(pk) = opt_val {
                        ctx.register_pk(table_idx, pk as i64, RowIdx(row_start + row_idx as u32));
                    }
                }
            }
        }
        DataType::UInt64 => {
            if let Ok(ca) = col.u64() {
                for (row_idx, opt_val) in ca.into_iter().enumerate() {
                    if let Some(pk) = opt_val {
                        ctx.register_pk(table_idx, pk as i64, RowIdx(row_start + row_idx as u32));
                    }
                }
            }
        }
        DataType::UInt32 => {
            if let Ok(ca) = col.u32() {
                for (row_idx, opt_val) in ca.into_iter().enumerate() {
                    if let Some(pk) = opt_val {
                        ctx.register_pk(table_idx, pk as i64, RowIdx(row_start + row_idx as u32));
                    }
                }
            }
        }
        _ => {}
    }
}

// ============================================================================
// Phase 4: Build FK Edges
// ============================================================================

fn build_fk_edges(
    table_infos: &[TableInfo],
    db: &mut Database,
    _column_name_to_idx: &HashMap<(TableIdx, String), ColumnIdx>,
    table_name_to_idx: &HashMap<String, TableIdx>,
    ctx: &PreprocessingContext,
) -> Result<()> {
    // Count FK columns for progress
    let total_fk_cols: usize = table_infos
        .iter()
        .filter_map(|info| info.metadata.as_ref())
        .map(|m| m.foreign_key_column_to_primary_key_table.len())
        .sum();

    if total_fk_cols == 0 {
        info!("No foreign key edges to build");
        // Initialize empty CSR
        db.build_csr_from_edges(Vec::new());
        return Ok(());
    }

    let pb = ProgressBar::new(total_fk_cols as u64);
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} Building FK edges: {pos}/{len} columns")
            .unwrap(),
    );

    // Collect all edges: (from_row, to_row)
    let mut all_edges: Vec<(u32, u32)> = Vec::new();

    for (table_idx, info) in table_infos.iter().enumerate() {
        let table_idx = TableIdx(table_idx as u32);

        if let Some(ref meta) = info.metadata {
            if meta.foreign_key_column_to_primary_key_table.is_empty() {
                continue;
            }

            let fk_col_names: Vec<&str> = meta
                .foreign_key_column_to_primary_key_table
                .keys()
                .map(|s| s.as_str())
                .collect();

            let df = LazyFrame::scan_parquet(
                PlPath::new(info.path.to_str().unwrap()),
                ScanArgsParquet::default(),
            )?
            .select(fk_col_names.iter().map(|s| col(*s)).collect::<Vec<_>>())
            .collect()
            .with_context(|| format!("Failed to read FK columns from: {:?}", info.path))?;

            let row_start = db.tables[table_idx.0 as usize].row_range.0.0;

            for (fk_col_name, target_table_name) in &meta.foreign_key_column_to_primary_key_table {
                if let Some(&target_table_idx) = table_name_to_idx.get(target_table_name) {
                    if let Ok(fk_col) = df.column(fk_col_name) {
                        collect_fk_edges_vectorized(
                            fk_col,
                            target_table_idx,
                            row_start,
                            ctx,
                            &mut all_edges,
                        );
                    }
                }
                pb.inc(1);
            }
        }
    }

    pb.finish_and_clear();
    info!("Created {} FK edges", all_edges.len());

    // Build CSR from edges
    db.build_csr_from_edges(all_edges);

    Ok(())
}

fn collect_fk_edges_vectorized(
    fk_col: &Column,
    target_table: TableIdx,
    row_start: u32,
    ctx: &PreprocessingContext,
    edges: &mut Vec<(u32, u32)>,
) {
    let mut add_edge = |opt_val: Option<i64>, row_idx: usize| {
        if let Some(fk_val) = opt_val {
            if let Some(target_row) = ctx.lookup_pk(target_table, fk_val) {
                edges.push((row_start + row_idx as u32, target_row.0));
            }
        }
    };

    match fk_col.dtype() {
        DataType::Int64 => {
            if let Ok(ca) = fk_col.i64() {
                for (row_idx, opt_val) in ca.into_iter().enumerate() {
                    add_edge(opt_val, row_idx);
                }
            }
        }
        DataType::Int32 => {
            if let Ok(ca) = fk_col.i32() {
                for (row_idx, opt_val) in ca.into_iter().enumerate() {
                    add_edge(opt_val.map(|v| v as i64), row_idx);
                }
            }
        }
        DataType::UInt64 => {
            if let Ok(ca) = fk_col.u64() {
                for (row_idx, opt_val) in ca.into_iter().enumerate() {
                    add_edge(opt_val.map(|v| v as i64), row_idx);
                }
            }
        }
        DataType::UInt32 => {
            if let Ok(ca) = fk_col.u32() {
                for (row_idx, opt_val) in ca.into_iter().enumerate() {
                    add_edge(opt_val.map(|v| v as i64), row_idx);
                }
            }
        }
        _ => {}
    }
}

// ============================================================================
// Phase 5: Embed Text Values
// ============================================================================

fn embed_text_values(db: &mut Database, ctx: &PreprocessingContext, embedder: &Embedder) {
    let pending_texts = &ctx.pending_texts;
    if pending_texts.is_empty() {
        return;
    }

    // Initialize embeddings storage
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
    info!("=== Phase 1: Collecting schema and statistics ===");
    let (table_infos, global_ts_stats) = collect_schema_and_stats(&args.input_dir, &metadata)?;
    info!(
        "Found {} tables, {} total rows",
        table_infos.len(),
        table_infos.iter().map(|t| t.num_rows).sum::<usize>()
    );

    // Phase 2: Build schema
    info!("=== Phase 2: Building schema ===");
    let (mut db, table_name_to_idx, column_name_to_idx) =
        build_schema(&table_infos, &embedder, &global_ts_stats);
    info!(
        "Schema: {} tables, {} columns",
        db.tables.len(),
        db.columns.len()
    );
    print_schema_summary(&table_infos);

    // Phase 3: Process tables
    info!("=== Phase 3: Processing tables ===");
    process_tables(&table_infos, &mut db, &column_name_to_idx, &mut ctx)?;
    info!(
        "Rows: {}, PK index: {} entries, Unique texts: {}",
        db.num_rows(),
        ctx.pk_index.len(),
        ctx.vocab_size()
    );

    // Phase 4: Build FK edges
    info!("=== Phase 4: Building FK edges ===");
    build_fk_edges(
        &table_infos,
        &mut db,
        &column_name_to_idx,
        &table_name_to_idx,
        &ctx,
    )?;

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
    // db.print_field_sizes();

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
