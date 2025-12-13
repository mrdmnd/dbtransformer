use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use half::f16;
use rkyv::{Archive, Deserialize, Serialize};

// ============================================================================
// Index Types (newtypes for type safety)
// ============================================================================

/// Global table index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, Hash, PartialEq, Eq))]
pub struct TableIdx(pub u32);

/// Global column index (across all tables)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, Hash, PartialEq, Eq))]
pub struct ColumnIdx(pub u32);

/// Global row index (across all tables)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, Hash, PartialEq, Eq))]
pub struct RowIdx(pub u32);

/// Index into the text vocabulary
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, Hash, PartialEq, Eq))]
pub struct TextIdx(pub u32);

// ============================================================================
// Semantic Types
// ============================================================================

/// Semantic type of a column - determines normalization and encoding strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[repr(u8)]
pub enum SemanticType {
    Number = 0,
    Text = 1,
    Datetime = 2,
    Boolean = 3,
}

// ============================================================================
// Table Type (for distinguishing DB tables from task tables)
// ============================================================================

/// Type of table - distinguishes regular database tables from task tables
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, Hash, PartialEq, Eq))]
pub enum TableType {
    /// Regular database table (the core relational data)
    #[default]
    Db,
    /// Task table - training split
    Train,
    /// Task table - validation split
    Val,
    /// Task table - test split
    Test,
}

// ============================================================================
// Cell Values
// ============================================================================

/// Raw cell value - preserves original data for temporal filtering, FK lookups, and debugging
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub enum RawCellValue {
    /// Raw numeric value (as read from source)
    Number(f32),
    /// Actual string content
    Text(String),
    /// Raw seconds since epoch (for temporal filtering)
    Datetime(f32),
    /// Raw boolean value
    Boolean(bool),
    /// Missing value
    Null,
}

/// Normalized cell value - ready for ML consumption
#[derive(Debug, Clone, Copy, PartialEq, Archive, Serialize, Deserialize)]
pub enum NormalizedCellValue {
    /// Z-score normalized scalar (for Number, Boolean, Datetime)
    Scalar(f32),
    /// Index into text value vocabulary (for embedding lookup)
    Text(TextIdx),
    /// Missing value (represented as NaN in practice, but explicit for clarity)
    Null,
}

// ============================================================================
// Schema: Column
// ============================================================================

/// Column metadata (all string references are via TextIdx)
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct Column {
    /// Column name
    pub name: String,
    /// Global column index
    pub idx: ColumnIdx,
    /// Local index within the table (0, 1, 2, ...)
    pub local_idx: u32,
    /// Which table this column belongs to
    pub table_idx: TableIdx,
    /// Schema string for embedding (index into vocab, e.g., "circuitId of circuits")
    pub schema_idx: TextIdx,
    /// Semantic type determines normalization strategy
    pub dtype: SemanticType,
    /// True if this is the primary key column
    pub is_primary_key: bool,
    /// If this is a foreign key, the table it references
    pub fk_target_table: Option<TableIdx>,
    /// If this is a foreign key, the column in the target table it references
    pub fk_target_column: Option<ColumnIdx>,
    // We store the frozen embedding of "<column_name> of <table_name>" directly on this object.
    pub column_description_embedding: Option<Vec<f16>>,
    /// Normalization mean (for Number/Boolean columns, computed per-column)
    pub norm_mean: Option<f32>,
    /// Normalization std (for Number/Boolean columns, computed per-column)
    pub norm_std: Option<f32>,
}

// ============================================================================
// Schema: Table
// ============================================================================

/// Table metadata (all string references are via TextIdx)
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct Table {
    /// Table name
    pub name: String,
    /// Global table index
    pub idx: TableIdx,
    /// Type of table (Db, Train, Val, Test)
    pub table_type: TableType,
    /// Range of column indices for this table [start, end)
    pub column_range: (ColumnIdx, ColumnIdx),
    /// Range of row indices for this table [start, end)
    pub row_range: (RowIdx, RowIdx),
    /// Primary key column (global index)
    pub primary_key_col: Option<ColumnIdx>,
    /// Time column for temporal queries (global index)
    pub time_col: Option<ColumnIdx>,
}

impl Table {
    /// Number of columns in this table
    pub fn num_columns(&self) -> u32 {
        self.column_range.1.0 - self.column_range.0.0
    }

    /// Number of rows in this table
    pub fn num_rows(&self) -> u32 {
        self.row_range.1.0 - self.row_range.0.0
    }
}

// ============================================================================
// Data: Row
// ============================================================================

/// A row containing both raw and normalized cell values
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct Row {
    /// Global row index
    pub idx: RowIdx,
    /// Which table this row belongs to
    pub table_idx: TableIdx,
    /// True if this row belongs to a task table (Train/Val/Test), false for Db tables
    pub is_task_row: bool,
    /// Raw cell values (for temporal filtering, FK lookups, debugging)
    pub raw: Vec<RawCellValue>,
    /// Normalized cell values (for ML consumption, populated by Database::normalize())
    pub normalized: Vec<NormalizedCellValue>,
}

// ============================================================================
// Foreign Key Edge
// ============================================================================

/// An edge in the database graph (FK relationship)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
pub struct ForeignKeyEdge {
    /// Row containing the foreign key
    pub from_row: RowIdx,
    /// Column containing the foreign key
    pub from_col: ColumnIdx,
    /// Row being referenced (in the target table)
    pub to_row: RowIdx,
}

// ============================================================================
// The Database
// ============================================================================

/// Complete database with schema, data, and vocabulary
#[derive(Debug, Archive, Serialize, Deserialize)]
pub struct Database {
    // --- Vocabulary (text value embeddings) ---
    /// Maps seen text values to their TextIdx
    pub text_value_lookup: HashMap<String, TextIdx>,
    /// Embeddings for each text value (indexed by TextIdx.0)
    pub text_value_embeddings: Vec<Vec<f16>>,

    // --- Schema ---
    /// All tables (indexed by TableIdx)
    pub tables: Vec<Table>,
    /// All columns (indexed by ColumnIdx)
    pub columns: Vec<Column>,

    // --- Data ---
    /// All rows (indexed by RowIdx)
    pub rows: Vec<Row>,

    // --- Graph Edges (FK relationships) ---
    /// All foreign key edges
    pub fk_edges: Vec<ForeignKeyEdge>,
    /// Outgoing edges from each row: row_idx -> [edge indices]
    pub edges_from: Vec<Vec<usize>>,
    /// Incoming edges to each row: row_idx -> [edge indices]
    pub edges_to: Vec<Vec<usize>>,

    // --- Primary Key Index ---
    /// Maps (table_idx, pk_value) -> row_idx for FK lookups
    /// pk_value is stored as i64 (works for int PKs)
    pub pk_index: HashMap<(TableIdx, i64), RowIdx>,

    // --- Temporal Range ---
    /// Minimum timestamp seen across all datetime columns (epoch seconds as f32)
    pub min_timestamp: Option<f32>,
    /// Maximum timestamp seen across all datetime columns (epoch seconds as f32)
    pub max_timestamp: Option<f32>,

    // --- Global Datetime Normalization ---
    /// Global mean of all datetime values (for z-score normalization)
    pub datetime_norm_mean: Option<f32>,
    /// Global std of all datetime values (for z-score normalization)
    pub datetime_norm_std: Option<f32>,
}

impl Database {
    /// Create a new empty database
    pub fn new() -> Self {
        Self {
            text_value_lookup: HashMap::new(),
            text_value_embeddings: Vec::new(),
            tables: Vec::new(),
            columns: Vec::new(),
            rows: Vec::new(),
            fk_edges: Vec::new(),
            edges_from: Vec::new(),
            edges_to: Vec::new(),
            pk_index: HashMap::new(),
            min_timestamp: None,
            max_timestamp: None,
            datetime_norm_mean: None,
            datetime_norm_std: None,
        }
    }

    /// Update the timestamp range with a new timestamp value
    pub fn update_timestamp_range(&mut self, timestamp: f32) {
        self.min_timestamp = Some(
            self.min_timestamp
                .map(|min| min.min(timestamp))
                .unwrap_or(timestamp),
        );
        self.max_timestamp = Some(
            self.max_timestamp
                .map(|max| max.max(timestamp))
                .unwrap_or(timestamp),
        );
    }

    /// Check if a row's time column value is before or at the given timestamp.
    /// Returns true if the row has no time column or if the time is <= cutoff.
    /// Uses raw datetime values (epoch seconds) for comparison.
    pub fn row_is_before(&self, row_idx: RowIdx, cutoff_timestamp: f32) -> bool {
        let row = self.get_row(row_idx);
        let table = self.get_table(row.table_idx);

        // If table has no time column, include the row
        let Some(time_col) = table.time_col else {
            return true;
        };

        // Get the local column index within the table
        let local_col_idx = (time_col.0 - table.column_range.0.0) as usize;

        // Check the raw cell value (uses raw epoch seconds, not normalized)
        match row.raw.get(local_col_idx) {
            Some(RawCellValue::Datetime(ts)) => *ts <= cutoff_timestamp,
            _ => true, // Include rows with null or missing time values
        }
    }

    /// Get or insert a text value, computing its embedding if new.
    /// Returns the TextIdx for this value.
    ///
    /// The `embed` closure is only called if this is a new string.
    pub fn intern_text<F>(&mut self, s: &str, embed: F) -> TextIdx
    where
        F: FnOnce(&str) -> Vec<f16>,
    {
        if let Some(&idx) = self.text_value_lookup.get(s) {
            idx
        } else {
            let idx = TextIdx(self.text_value_lookup.len() as u32);
            self.text_value_lookup.insert(s.to_string(), idx);
            self.text_value_embeddings.push(embed(s));
            idx
        }
    }

    /// Get a table by index
    pub fn get_table(&self, idx: TableIdx) -> &Table {
        &self.tables[idx.0 as usize]
    }

    /// Get a column by index
    pub fn get_column(&self, idx: ColumnIdx) -> &Column {
        &self.columns[idx.0 as usize]
    }

    /// Get a row by index
    pub fn get_row(&self, idx: RowIdx) -> &Row {
        &self.rows[idx.0 as usize]
    }

    /// Get table name as string
    pub fn table_name(&self, idx: TableIdx) -> &str {
        &self.get_table(idx).name
    }

    /// Get column name as string
    pub fn column_name(&self, idx: ColumnIdx) -> &str {
        &self.get_column(idx).name
    }

    /// Number of tables
    pub fn num_tables(&self) -> usize {
        self.tables.len()
    }

    /// Number of columns
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Number of rows
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Number of unique text values
    pub fn vocab_size(&self) -> usize {
        self.text_value_lookup.len()
    }

    /// Save the database to a file using rkyv serialization.
    ///
    /// This creates a compact binary representation that can be quickly loaded back.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(self).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Serialization error: {e}"),
            )
        })?;
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&bytes)?;
        writer.flush()?;
        Ok(())
    }

    /// Load a database from a file using rkyv deserialization.
    ///
    /// This reads the binary representation and reconstructs the full Database.
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        let db = rkyv::from_bytes::<Self, rkyv::rancor::Error>(&bytes).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Deserialization error: {e}"),
            )
        })?;
        Ok(db)
    }

    /// Normalize all raw cell values and populate the normalized vectors.
    ///
    /// This computes:
    /// - Per-column z-score normalization for Number and Boolean columns
    /// - Global z-score normalization for all Datetime values
    /// - TextIdx references for Text values (pointing to embedding table)
    ///
    /// Should be called after all raw data is loaded but before saving.
    pub fn normalize(&mut self) {
        // First pass: collect statistics
        // - Per-column: sum and count for Number/Boolean
        // - Global: sum and count for all Datetime values
        let num_cols = self.columns.len();
        let mut col_sums: Vec<f64> = vec![0.0; num_cols];
        let mut col_counts: Vec<u64> = vec![0; num_cols];
        let mut datetime_sum: f64 = 0.0;
        let mut datetime_count: u64 = 0;

        for row in &self.rows {
            let table = &self.tables[row.table_idx.0 as usize];
            let col_start = table.column_range.0.0 as usize;

            for (local_idx, cell) in row.raw.iter().enumerate() {
                let col_idx = col_start + local_idx;
                let col = &self.columns[col_idx];

                match (cell, col.dtype) {
                    (RawCellValue::Number(v), SemanticType::Number) => {
                        col_sums[col_idx] += *v as f64;
                        col_counts[col_idx] += 1;
                    }
                    (RawCellValue::Boolean(v), SemanticType::Boolean) => {
                        col_sums[col_idx] += if *v { 1.0 } else { 0.0 };
                        col_counts[col_idx] += 1;
                    }
                    (RawCellValue::Datetime(v), SemanticType::Datetime) => {
                        datetime_sum += *v as f64;
                        datetime_count += 1;
                    }
                    _ => {}
                }
            }
        }

        // Compute means
        let col_means: Vec<f64> = col_sums
            .iter()
            .zip(col_counts.iter())
            .map(|(sum, count)| if *count > 0 { sum / *count as f64 } else { 0.0 })
            .collect();

        let datetime_mean = if datetime_count > 0 {
            datetime_sum / datetime_count as f64
        } else {
            0.0
        };

        // Second pass: compute variance
        let mut col_var_sums: Vec<f64> = vec![0.0; num_cols];
        let mut datetime_var_sum: f64 = 0.0;

        for row in &self.rows {
            let table = &self.tables[row.table_idx.0 as usize];
            let col_start = table.column_range.0.0 as usize;

            for (local_idx, cell) in row.raw.iter().enumerate() {
                let col_idx = col_start + local_idx;
                let col = &self.columns[col_idx];

                match (cell, col.dtype) {
                    (RawCellValue::Number(v), SemanticType::Number) => {
                        let diff = *v as f64 - col_means[col_idx];
                        col_var_sums[col_idx] += diff * diff;
                    }
                    (RawCellValue::Boolean(v), SemanticType::Boolean) => {
                        let val = if *v { 1.0 } else { 0.0 };
                        let diff = val - col_means[col_idx];
                        col_var_sums[col_idx] += diff * diff;
                    }
                    (RawCellValue::Datetime(v), SemanticType::Datetime) => {
                        let diff = *v as f64 - datetime_mean;
                        datetime_var_sum += diff * diff;
                    }
                    _ => {}
                }
            }
        }

        // Compute stds (with minimum of 1e-8 to avoid division by zero)
        let col_stds: Vec<f64> = col_var_sums
            .iter()
            .zip(col_counts.iter())
            .map(|(var_sum, count)| {
                if *count > 1 {
                    (var_sum / (*count - 1) as f64).sqrt().max(1e-8)
                } else {
                    1.0 // Single value or no values: use std=1 to avoid NaN
                }
            })
            .collect();

        let datetime_std = if datetime_count > 1 {
            (datetime_var_sum / (datetime_count - 1) as f64)
                .sqrt()
                .max(1e-8)
        } else {
            1.0
        };

        // Store normalization parameters
        self.datetime_norm_mean = Some(datetime_mean as f32);
        self.datetime_norm_std = Some(datetime_std as f32);

        for (col_idx, col) in self.columns.iter_mut().enumerate() {
            match col.dtype {
                SemanticType::Number | SemanticType::Boolean => {
                    col.norm_mean = Some(col_means[col_idx] as f32);
                    col.norm_std = Some(col_stds[col_idx] as f32);
                }
                _ => {
                    col.norm_mean = None;
                    col.norm_std = None;
                }
            }
        }

        // Third pass: populate normalized vectors
        for row in &mut self.rows {
            let table = &self.tables[row.table_idx.0 as usize];
            let col_start = table.column_range.0.0 as usize;

            row.normalized = row
                .raw
                .iter()
                .enumerate()
                .map(|(local_idx, cell)| {
                    let col_idx = col_start + local_idx;
                    let col = &self.columns[col_idx];

                    match (cell, col.dtype) {
                        (RawCellValue::Number(v), SemanticType::Number) => {
                            let z = (*v as f64 - col_means[col_idx]) / col_stds[col_idx];
                            NormalizedCellValue::Scalar(z as f32)
                        }
                        (RawCellValue::Boolean(v), SemanticType::Boolean) => {
                            let val = if *v { 1.0 } else { 0.0 };
                            let z = (val - col_means[col_idx]) / col_stds[col_idx];
                            NormalizedCellValue::Scalar(z as f32)
                        }
                        (RawCellValue::Datetime(v), SemanticType::Datetime) => {
                            let z = (*v as f64 - datetime_mean) / datetime_std;
                            NormalizedCellValue::Scalar(z as f32)
                        }
                        (RawCellValue::Text(s), SemanticType::Text) => {
                            // Look up the TextIdx for this string
                            if let Some(&idx) = self.text_value_lookup.get(s) {
                                NormalizedCellValue::Text(idx)
                            } else {
                                // This shouldn't happen if data was loaded correctly
                                NormalizedCellValue::Null
                            }
                        }
                        (RawCellValue::Null, _) => NormalizedCellValue::Null,
                        _ => NormalizedCellValue::Null, // Type mismatch fallback
                    }
                })
                .collect();
        }
    }

    /// Verbose debug dump of the entire database structure
    pub fn dump_verbose(&self) {
        println!(
            "╔══════════════════════════════════════════════════════════════════════════════╗"
        );
        println!(
            "║                           DATABASE VERBOSE DUMP                              ║"
        );
        println!(
            "╚══════════════════════════════════════════════════════════════════════════════╝"
        );
        println!();

        // Summary
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ SUMMARY                                                                     │");
        println!("├─────────────────────────────────────────────────────────────────────────────┤");
        println!(
            "│ Tables:              {:>10}                                            │",
            self.tables.len()
        );
        println!(
            "│ Columns:             {:>10}                                            │",
            self.columns.len()
        );
        println!(
            "│ Rows:                {:>10}                                            │",
            self.rows.len()
        );
        println!(
            "│ Vocabulary size:     {:>10}                                            │",
            self.vocab_size()
        );
        println!(
            "│ FK edges:            {:>10}                                            │",
            self.fk_edges.len()
        );
        println!(
            "│ PK index entries:    {:>10}                                            │",
            self.pk_index.len()
        );
        println!(
            "│ Min timestamp:       {:>10.0}                                            │",
            self.min_timestamp.unwrap_or(f32::NAN)
        );
        println!(
            "│ Max timestamp:       {:>10.0}                                            │",
            self.max_timestamp.unwrap_or(f32::NAN)
        );
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        println!();

        // Tables
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ TABLES                                                                      │");
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        for table in &self.tables {
            println!("  Table[{}]: \"{}\"", table.idx.0, table.name);
            println!(
                "    column_range: [{}, {})",
                table.column_range.0.0, table.column_range.1.0
            );
            println!(
                "    row_range:    [{}, {})",
                table.row_range.0.0, table.row_range.1.0
            );
            println!("    num_columns:  {}", table.num_columns());
            println!("    num_rows:     {}", table.num_rows());
            println!("    primary_key:  {:?}", table.primary_key_col.map(|c| c.0));
            println!("    time_col:     {:?}", table.time_col.map(|c| c.0));
            println!();
        }

        // Columns
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ COLUMNS                                                                     │");
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        for col in &self.columns {
            let table_name = &self.tables[col.table_idx.0 as usize].name;
            println!("  Column[{}]: \"{}.{}\"", col.idx.0, table_name, col.name);
            println!("    local_idx:    {}", col.local_idx);
            println!("    table_idx:    {}", col.table_idx.0);
            println!("    schema_idx:   {}", col.schema_idx.0);
            println!("    dtype:        {:?}", col.dtype);
            println!("    is_pk:        {}", col.is_primary_key);
            println!(
                "    fk_target:    {:?} -> {:?}",
                col.fk_target_table.map(|t| t.0),
                col.fk_target_column.map(|c| c.0)
            );
            println!(
                "    embedding:    {} dims",
                col.column_description_embedding
                    .as_ref()
                    .map(|e| e.len())
                    .unwrap_or(0)
            );
            println!();
        }

        // Vocabulary (first 50 entries)
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!(
            "│ VOCABULARY (first 50 of {})                                   │",
            self.vocab_size()
        );
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        let mut vocab_items: Vec<_> = self.text_value_lookup.iter().collect();
        vocab_items.sort_by_key(|(_, idx)| idx.0);
        for (text, idx) in vocab_items.iter().take(50) {
            let embedding_len = self
                .text_value_embeddings
                .get(idx.0 as usize)
                .map(|e| e.len())
                .unwrap_or(0);
            let text_preview: String = text.chars().take(60).collect();
            let truncated = if text.len() > 60 { "..." } else { "" };
            println!(
                "  TextIdx[{}]: \"{}{}\", embedding: {} dims",
                idx.0, text_preview, truncated, embedding_len
            );
        }
        if self.vocab_size() > 50 {
            println!("  ... and {} more entries", self.vocab_size() - 50);
        }
        println!();

        // Rows (first 20 entries)
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!(
            "│ ROWS (first 20 of {})                                          │",
            self.rows.len()
        );
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        for row in self.rows.iter().take(20) {
            let table_name = &self.tables[row.table_idx.0 as usize].name;
            println!("  Row[{}]: table=\"{}\"", row.idx.0, table_name);
            print!("    raw: [");
            for (i, cell) in row.raw.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                match cell {
                    RawCellValue::Number(v) => print!("Num({:.2})", v),
                    RawCellValue::Text(s) => {
                        let preview: String = s.chars().take(15).collect();
                        let trunc = if s.len() > 15 { ".." } else { "" };
                        print!("Txt(\"{}{}\")", preview, trunc)
                    }
                    RawCellValue::Datetime(v) => print!("Dt({:.0})", v),
                    RawCellValue::Boolean(v) => print!("Bool({})", v),
                    RawCellValue::Null => print!("Null"),
                }
            }
            println!("]");
            print!("    norm: [");
            for (i, cell) in row.normalized.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                match cell {
                    NormalizedCellValue::Scalar(v) => print!("{:.3}", v),
                    NormalizedCellValue::Text(idx) => print!("T{}", idx.0),
                    NormalizedCellValue::Null => print!("NaN"),
                }
            }
            println!("]");
        }
        if self.rows.len() > 20 {
            println!("  ... and {} more rows", self.rows.len() - 20);
        }
        println!();

        // FK Edges (first 30)
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!(
            "│ FK EDGES (first 30 of {})                                       │",
            self.fk_edges.len()
        );
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        for (i, edge) in self.fk_edges.iter().take(30).enumerate() {
            let from_row = &self.rows[edge.from_row.0 as usize];
            let to_row = &self.rows[edge.to_row.0 as usize];
            let from_table = &self.tables[from_row.table_idx.0 as usize].name;
            let to_table = &self.tables[to_row.table_idx.0 as usize].name;
            let col_name = &self.columns[edge.from_col.0 as usize].name;
            println!(
                "  Edge[{}]: {}.Row[{}].{} -> {}.Row[{}]",
                i, from_table, edge.from_row.0, col_name, to_table, edge.to_row.0
            );
        }
        if self.fk_edges.len() > 30 {
            println!("  ... and {} more edges", self.fk_edges.len() - 30);
        }
        println!();

        // Edge adjacency stats
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ EDGE ADJACENCY STATS                                                        │");
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        let total_from: usize = self.edges_from.iter().map(|v| v.len()).sum();
        let total_to: usize = self.edges_to.iter().map(|v| v.len()).sum();
        let max_from = self.edges_from.iter().map(|v| v.len()).max().unwrap_or(0);
        let max_to = self.edges_to.iter().map(|v| v.len()).max().unwrap_or(0);
        let non_empty_from = self.edges_from.iter().filter(|v| !v.is_empty()).count();
        let non_empty_to = self.edges_to.iter().filter(|v| !v.is_empty()).count();
        println!(
            "  edges_from: {} lists, {} total edges, max degree {}, {} non-empty",
            self.edges_from.len(),
            total_from,
            max_from,
            non_empty_from
        );
        println!(
            "  edges_to:   {} lists, {} total edges, max degree {}, {} non-empty",
            self.edges_to.len(),
            total_to,
            max_to,
            non_empty_to
        );
        println!();

        // PK Index (first 20)
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!(
            "│ PK INDEX (first 20 of {})                                       │",
            self.pk_index.len()
        );
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        let mut pk_items: Vec<_> = self.pk_index.iter().collect();
        pk_items.sort_by_key(|((t, pk), _)| (t.0, *pk));
        for ((table_idx, pk_value), row_idx) in pk_items.iter().take(20) {
            let table_name = &self.tables[table_idx.0 as usize].name;
            println!("  ({}, pk={}) -> Row[{}]", table_name, pk_value, row_idx.0);
        }
        if self.pk_index.len() > 20 {
            println!("  ... and {} more entries", self.pk_index.len() - 20);
        }
        println!();

        // Memory estimates
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ MEMORY ESTIMATES                                                            │");
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        let embedding_bytes: usize = self
            .text_value_embeddings
            .iter()
            .map(|e| e.len() * 2) // f16 = 2 bytes
            .sum();
        let col_embedding_bytes: usize = self
            .columns
            .iter()
            .filter_map(|c| c.column_description_embedding.as_ref())
            .map(|e| e.len() * 2)
            .sum();
        let cell_count: usize = self.rows.iter().map(|r| r.raw.len()).sum();
        println!(
            "  Text embeddings:   {:>10} bytes ({:.2} MB)",
            embedding_bytes,
            embedding_bytes as f64 / 1_048_576.0
        );
        println!(
            "  Column embeddings: {:>10} bytes ({:.2} MB)",
            col_embedding_bytes,
            col_embedding_bytes as f64 / 1_048_576.0
        );
        println!("  Total cells:       {:>10}", cell_count);
        println!();

        println!(
            "╔══════════════════════════════════════════════════════════════════════════════╗"
        );
        println!(
            "║                           END OF VERBOSE DUMP                                ║"
        );
        println!(
            "╚══════════════════════════════════════════════════════════════════════════════╝"
        );
    }
}

impl Default for Database {
    fn default() -> Self {
        Self::new()
    }
}
