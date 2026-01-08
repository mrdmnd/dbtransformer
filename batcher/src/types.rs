use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use half::f16;
use rkyv::{Archive, Deserialize, Serialize};

// ============================================================================
// Index Types
// ============================================================================

/// Global table index in the database
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, Hash, PartialEq, Eq))]
pub struct TableIdx(pub usize);

/// Global column index (unique across all tables) in the database
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, Hash, PartialEq, Eq))]
pub struct ColumnIdx(pub usize);

/// Global row index (unique across all tables) in the database
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, Hash, PartialEq, Eq))]
pub struct RowIdx(pub usize);

/// Index into an 'interned' text vocabulary.
/// All of the text values in the DB cells get embedded by a frozen embedding model, and stored here.
/// We don't do this for the column description embeddings though - those are stored directly on the Column object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, Hash, PartialEq, Eq))]
pub struct TextIdx(pub usize);

// ============================================================================
// Semantic Types
// ============================================================================

/// Semantic type of a database column - determines normalization and encoding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[repr(u8)]
pub enum SemanticType {
    // Integers and floats, where numerical ordering is meaningful
    // e.g. product price, age, discount percentage, etc.
    // Prediction task is REGRESSION
    // Polars Float{16, 32, 64} and Decimal are auto-detected into this stype.
    // Polars Int{8, 16, 32, 64, 128} are auto-detected into this stype, unless explicitly marked as Categorical in the metadata.
    // Polars Duration is auto-detected into this stype.
    // I'm not actually sure what do do about polars UInt types.
    Numerical = 0,

    // Strings, integers, or booleans, with a limited number of unique values.
    // e.g. product type, subscription status, etc.
    // Prediction task is CLASSIFICATION
    // Polars dtype Boolean, Categorical, Enum are auto-detected into this type.
    // Polars String and Int{8, 16, 32, 64, 128} are NOT auto-detected into this type, unless explicitly marked as Categorical in the metadata.
    Categorical = 1,

    // Date/time values
    // Use this instead of "Numerical" for values that are specifically timestamps.
    // Prediction task is REGRESSION
    // Polars Date, Datetime, and Time are auto-detected into this stype.
    // The other Polars types are NOT auto-detected into this stype.
    // Notably, "Duration" is turned into a Numerical type.
    Timestamp = 2,

    // Multi-token strings where semantic meaning is important
    // e.g. product description, customer review, etc.
    // If the column is *stored* as text, but represents a categorical value, prefer to use the Categorical type.
    // No prediction task is associated with this type... YET...
    Text = 3,

    // All other Polars dtypes are unsupported..
    // Binary, Extension, ...
    Unsupported = 5,
}

// ============================================================================
// Cell Values
// ============================================================================

// For loading into an ML model, we want to normalize our cell values.
// For numerical values, we'll use z-score normalization, using the column-specific mean and std.
// For categorical values, we'll ... not sure?
// For timestamp values, we'll also use z-score normalization, but we'll use the global mean and std for timestamps.
// Identifier values are not normalized.
// For text values, we'll use a frozen embedding model to embed the text into some fixed-size vector.

#[derive(Debug, Clone, Copy, PartialEq, Archive, Serialize, Deserialize)]
pub enum NormalizedCellValue {
    /// Z-score normalized scalar (for Nu
    Numerical(f32),
    /// Index into text value vocabulary (for embedding lookup)
    Text(TextIdx),
    /// Missing value (rexplicit for clarity right now)
    /// Note - this is different from floating point NaN.
    Null,
}

// ============================================================================
// Schema: Column
// ============================================================================

/// Column metadata
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct Column {
    /// Column name
    pub name: String,

    /// Global column index in the database
    pub idx: ColumnIdx,

    /// Which table this column belongs to
    pub table_idx: TableIdx,

    /// Semantic type determines normalization strategy
    pub dtype: SemanticType,

    /// True if this is the primary key column for the parent table
    pub is_primary_key: bool,

    /// If this is a foreign key, the column in the target table it references
    pub fk_target_column: Option<ColumnIdx>,

    // We store the frozen embedding of "<column_name> of <table_name>" directly on this object.
    // At creation time; it may not be available yet, but we create these Column objects as `mut` and let the embedder
    // assign the vector later.
    pub column_description_embedding: Option<Vec<f16>>,

    /// Normalization mean (for Number/Boolean columns, computed per-column)
    /// Assigned during preprocessing once column statistics are known.
    pub norm_mean: Option<f32>,

    /// Normalization std (for Number/Boolean columns, computed per-column)
    /// Assigned during preprocessing once column statistics are known.
    pub norm_std: Option<f32>,
    // TODO(mrdmnd): consider computing online mean and variance as we go? Then we can skip the normalization secondary
    // pass.
}

// ============================================================================
// Schema: Table
// ============================================================================

/// Table metadata (all string references are via TextIdx)
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct Table {
    /// Table name
    pub name: String,

    /// Global table index in the database
    pub idx: TableIdx,

    /// Range of column indices for this table [start, end)
    pub column_range: (ColumnIdx, ColumnIdx),

    /// Range of row indices for this table [start, end)
    pub row_range: (RowIdx, RowIdx),

    /// Primary key column (global index)
    pub primary_key_col: Option<ColumnIdx>,

    /// Reference time column for temporal queries (global index)
    pub time_col: Option<ColumnIdx>,
}

impl Table {
    /// Number of columns in this table
    pub fn num_columns(&self) -> usize {
        self.column_range.1.0 - self.column_range.0.0
    }

    /// Number of rows in this table
    pub fn num_rows(&self) -> usize {
        self.row_range.1.0 - self.row_range.0.0
    }
}

/// A post-processed row contains normalized cell values.
/// For sampling purposes, we'll also want to keep the *raw* values for the time column, if present.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct Row {
    /// Global row index
    pub idx: RowIdx,
    /// Which table this row belongs to
    pub table_idx: TableIdx,

    /// Raw timestamp for the table's time column (epoch seconds), if present.
    pub raw_timestamp: Option<f32>,

    /// Normalized cell values (for ML consumption)
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
    /// Maps discovered text values to their TextIdx
    /// Think about this like an "interned" string table.
    pub text_value_lookup: HashMap<String, TextIdx>,

    /// Embeddings for each text value (indexed by TextIdx.0)
    /// Will have a number of rows equal to the number of unique text values in the database.
    pub text_value_embeddings: Vec<Vec<f16>>,

    // --- Schema / metadata ---
    pub tables: Vec<Table>,
    pub columns: Vec<Column>,

    // --- Data ---
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

        // Prefer precomputed raw timestamp if available
        if let Some(ts) = row.raw_timestamp {
            return ts <= cutoff_timestamp;
        }

        // Get the local column index within the table
        let local_col_idx = (time_col.0 - table.column_range.0.0) as usize;

        if let Some(NormalizedCellValue::Scalar(z)) = row.normalized.get(local_col_idx) {
            let mean = self.datetime_norm_mean.unwrap_or(0.0);
            let std = self.datetime_norm_std.unwrap_or(1.0).max(1e-8);
            let ts = *z * std + mean;
            ts <= cutoff_timestamp
        } else {
            true
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

    /// Build adjacency lists (outgoing and incoming) from fk_edges.
    pub fn build_adjacency(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut edges_from: Vec<Vec<usize>> = vec![Vec::new(); self.rows.len()];
        let mut edges_to: Vec<Vec<usize>> = vec![Vec::new(); self.rows.len()];

        for (edge_idx, edge) in self.fk_edges.iter().enumerate() {
            edges_from[edge.from_row.0 as usize].push(edge_idx);
            edges_to[edge.to_row.0 as usize].push(edge_idx);
        }

        (edges_from, edges_to)
    }

    /// Rebuild and store adjacency lists from fk_edges.
    pub fn rebuild_adjacency(&mut self) {
        let (edges_from, edges_to) = self.build_adjacency();
        self.edges_from = edges_from;
        self.edges_to = edges_to;
    }

    /// Ensure adjacency lists exist and are sized correctly.
    pub fn ensure_adjacency(&mut self) {
        if self.edges_from.len() != self.rows.len() || self.edges_to.len() != self.rows.len() {
            self.rebuild_adjacency();
        }
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
        let mut db = rkyv::from_bytes::<Self, rkyv::rancor::Error>(&bytes).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Deserialization error: {e}"),
            )
        })?;
        db.ensure_adjacency();
        Ok(db)
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
            println!("    table_idx:    {}", col.table_idx.0);
            println!("    dtype:        {:?}", col.dtype);
            println!("    is_pk:        {}", col.is_primary_key);
            println!("    fk_target:    {:?}", col.fk_target_column.map(|c| c.0));
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
            println!(
                "  Row[{}]: table=\"{}\", raw_timestamp={:?}",
                row.idx.0, table_name, row.raw_timestamp
            );
            print!("    values: [");
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
        if self.edges_from.len() == self.rows.len() && self.edges_to.len() == self.rows.len() {
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
        } else {
            println!(
                "  fk_edges:   {} total edges (adjacency not materialized in struct)",
                self.fk_edges.len()
            );
        }
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
        let cell_count: usize = self.rows.iter().map(|r| r.normalized.len()).sum();
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
