//! Types for representing a relational database as a graph structure.
//!
//! This module provides:
//! - Metadata types for loading database schema from JSON
//! - Schema types (Table, Column) with semantic type information
//! - Normalized cell values ready for ML model consumption
//! - Graph representation via foreign key edges
//!
//! The preprocessing pipeline transforms raw parquet files + metadata.json into
//! a `Database` struct that can be serialized via rkyv for fast loading.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use half::f16;
use rkyv::{Archive, Deserialize, Serialize};
use serde::Deserialize as SerdeDeserialize;

// ============================================================================
// Index Types (Newtype wrappers for type safety)
// ============================================================================

/// Global table index in the database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, Hash, PartialEq, Eq))]
pub struct TableIdx(pub usize);

/// Global column index (unique across all tables) in the database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, Hash, PartialEq, Eq))]
pub struct ColumnIdx(pub usize);

/// Global row index (unique across all tables) in the database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, Hash, PartialEq, Eq))]
pub struct RowIdx(pub usize);

/// Index into the interned text embedding vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, Hash, PartialEq, Eq))]
pub struct EmbeddingIdx(pub usize);

// ============================================================================
// Metadata Types (for loading from metadata.json)
// ============================================================================

/// Per-table metadata loaded from metadata.json.
///
/// Example JSON:
/// ```json
/// {
///     "primary_key_column": "id",
///     "foreign_key_column_to_primary_key_table": {"user_id": "user"},
///     "time_column": "created_at",
///     "stype_overrides": {"status": "categorical"},
///     "ignored_columns": ["internal_notes"]
/// }
/// ```
#[derive(Debug, Clone, SerdeDeserialize)]
pub struct TableMetadata {
    /// The column that serves as the primary key for this table.
    pub primary_key_column: Option<String>,

    /// Maps FK column names to the table they reference.
    /// e.g., {"user_id": "user", "movie_id": "movie"}
    #[serde(default)]
    pub foreign_key_column_to_primary_key_table: HashMap<String, String>,

    /// Optional timestamp column indicating when each row's data became valid.
    pub time_column: Option<String>,

    /// Override the auto-detected semantic type for specific columns.
    /// Valid values: "numerical", "categorical", "timestamp", "text"
    #[serde(default)]
    pub stype_overrides: HashMap<String, String>,

    /// Columns to exclude from processing entirely.
    #[serde(default)]
    pub ignored_columns: Vec<String>,
}

/// Complete database metadata: table name -> table metadata.
pub type DatabaseMetadata = HashMap<String, TableMetadata>;

/// Load database metadata from a metadata.json file.
pub fn load_metadata<P: AsRef<Path>>(path: P) -> std::io::Result<DatabaseMetadata> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to parse metadata.json: {e}"),
        )
    })
}

// ============================================================================
// Semantic Types
// ============================================================================

/// Semantic type of a database column - determines normalization and encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[repr(u8)]
pub enum SemanticType {
    /// Numeric values where ordering is meaningful (price, age, quantity).
    /// Normalized via z-score: (value - mean) / std
    Numerical = 0,

    /// Discrete values from a finite set (status, type, category).
    /// Embedded as "<column_name> is <value>" text.
    Categorical = 1,

    /// Date/time values with cyclical components.
    /// Encoded as sin/cos pairs for minute, hour, day, month, plus z-scored epoch.
    Timestamp = 2,

    /// Free-form text with semantic meaning (description, comment).
    /// Embedded via a frozen text embedding model.
    Text = 3,
}

impl SemanticType {
    /// Parse a semantic type from a string (for stype_overrides).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "numerical" | "numeric" | "number" => Some(Self::Numerical),
            "categorical" | "category" => Some(Self::Categorical),
            "timestamp" | "datetime" | "time" => Some(Self::Timestamp),
            "text" | "string" => Some(Self::Text),
            _ => None,
        }
    }
}

// ============================================================================
// Schema: Column
// ============================================================================

/// Column metadata and normalization parameters.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct Column {
    /// Column name as it appears in the source data.
    pub name: String,

    /// Global column index in the database.
    pub idx: ColumnIdx,

    /// Which table this column belongs to.
    pub table_idx: TableIdx,

    /// Semantic type determining how values are normalized/encoded.
    pub stype: SemanticType,

    /// True if this column is the primary key for its table.
    pub is_primary_key: bool,

    /// If this is a foreign key, the target column it references (always a PK).
    pub fk_target_column: Option<ColumnIdx>,

    /// Frozen embedding of "<column_name> of <table_name>" for model input.
    pub description_embedding: Vec<f16>,

    /// For Numerical columns: z-score normalization parameters.
    pub norm_mean: Option<f32>,
    pub norm_std: Option<f32>,

    /// For Categorical columns: the set of possible values.
    pub category_values: Option<Vec<String>>,
}

// ============================================================================
// Schema: Table
// ============================================================================

/// Table metadata with column and row ranges.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct Table {
    /// Table name as it appears in the source data.
    pub name: String,

    /// Global table index in the database.
    pub idx: TableIdx,

    /// Range of column indices for this table [start, end).
    pub column_range: (ColumnIdx, ColumnIdx),

    /// Range of row indices for this table [start, end).
    pub row_range: (RowIdx, RowIdx),

    /// The primary key column for this table (if any).
    pub primary_key_column: Option<ColumnIdx>,

    /// The timestamp column for temporal ordering (if any).
    pub time_column: Option<ColumnIdx>,
}

impl Table {
    /// Number of columns in this table.
    #[inline]
    pub fn num_columns(&self) -> usize {
        self.column_range.1 .0 - self.column_range.0 .0
    }

    /// Number of rows in this table.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.row_range.1 .0 - self.row_range.0 .0
    }

    /// Check if a column index belongs to this table.
    #[inline]
    pub fn contains_column(&self, col: ColumnIdx) -> bool {
        col.0 >= self.column_range.0 .0 && col.0 < self.column_range.1 .0
    }

    /// Check if a row index belongs to this table.
    #[inline]
    pub fn contains_row(&self, row: RowIdx) -> bool {
        row.0 >= self.row_range.0 .0 && row.0 < self.row_range.1 .0
    }
}

// ============================================================================
// Cell Values (Normalized for ML)
// ============================================================================

/// Timestamp encoding dimension: 5 cyclical pairs (sin/cos) + 1 linear = 11.
pub const TIMESTAMP_DIM: usize = 11;

/// A preprocessed cell value ready for model consumption.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub enum CellValue {
    /// Z-score normalized scalar: (value - mean) / std
    Numerical(f32),

    /// Index into the embedding vocabulary for categorical values.
    Categorical(EmbeddingIdx),

    /// Cyclical timestamp encoding (11 dimensions):
    /// - [0-1]  minute_of_hour (sin, cos)
    /// - [2-3]  hour_of_day (sin, cos)
    /// - [4-5]  day_of_week (sin, cos)
    /// - [6-7]  day_of_year (sin, cos)
    /// - [8-9]  month_of_year (sin, cos)
    /// - [10]   z-scored epoch seconds
    Timestamp([f32; TIMESTAMP_DIM]),

    /// Index into the embedding vocabulary for free-form text.
    Text(EmbeddingIdx),

    /// Missing/null value - excluded from attention.
    Null,
}

impl CellValue {
    /// Create a timestamp encoding from epoch seconds.
    pub fn from_epoch_seconds(epoch_secs: f64, mean: f64, std: f64) -> Self {
        use std::f64::consts::TAU;

        const SECS_PER_MINUTE: f64 = 60.0;
        const SECS_PER_HOUR: f64 = 3600.0;
        const SECS_PER_DAY: f64 = 86400.0;
        const DAYS_PER_YEAR: f64 = 365.25;

        let days_since_epoch = epoch_secs / SECS_PER_DAY;
        let secs_today = epoch_secs.rem_euclid(SECS_PER_DAY);

        let minute_of_hour = secs_today.rem_euclid(SECS_PER_HOUR) / SECS_PER_MINUTE;
        let hour_of_day = secs_today / SECS_PER_HOUR;
        let day_of_week = (days_since_epoch + 4.0).rem_euclid(7.0); // Jan 1 1970 = Thursday
        let day_of_year = days_since_epoch.rem_euclid(DAYS_PER_YEAR);
        let month = day_of_year / (DAYS_PER_YEAR / 12.0);

        let minute_angle = TAU * minute_of_hour / 60.0;
        let hour_angle = TAU * hour_of_day / 24.0;
        let dow_angle = TAU * day_of_week / 7.0;
        let doy_angle = TAU * day_of_year / DAYS_PER_YEAR;
        let month_angle = TAU * month / 12.0;

        let std_safe = std.max(1e-8);
        let epoch_zscore = (epoch_secs - mean) / std_safe;

        CellValue::Timestamp([
            minute_angle.sin() as f32,
            minute_angle.cos() as f32,
            hour_angle.sin() as f32,
            hour_angle.cos() as f32,
            dow_angle.sin() as f32,
            dow_angle.cos() as f32,
            doy_angle.sin() as f32,
            doy_angle.cos() as f32,
            month_angle.sin() as f32,
            month_angle.cos() as f32,
            epoch_zscore as f32,
        ])
    }

    /// Extract the z-scored epoch value from a Timestamp cell.
    pub fn epoch_zscore(&self) -> Option<f32> {
        match self {
            CellValue::Timestamp(arr) => Some(arr[TIMESTAMP_DIM - 1]),
            _ => None,
        }
    }

    /// Check if this cell is null.
    pub fn is_null(&self) -> bool {
        matches!(self, CellValue::Null)
    }
}

// ============================================================================
// Row
// ============================================================================

/// A single row in the database with all cell values.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct Row {
    /// Global row index.
    pub idx: RowIdx,

    /// Which table this row belongs to.
    pub table_idx: TableIdx,

    /// Cell values for each column (same order as columns in the table).
    pub values: Vec<CellValue>,
}

// ============================================================================
// Foreign Key Edge
// ============================================================================

/// An edge in the database graph representing a foreign key relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
pub struct ForeignKeyEdge {
    /// The row containing the foreign key value.
    pub from_row: RowIdx,

    /// The column containing the foreign key value.
    pub from_col: ColumnIdx,

    /// The row being referenced (in another table).
    pub to_row: RowIdx,
}

// ============================================================================
// Database
// ============================================================================

/// A complete preprocessed database ready for sampling.
///
/// Contains:
/// - Schema: tables and columns with metadata
/// - Data: rows with normalized cell values
/// - Graph: foreign key edges with adjacency lists
/// - Vocabulary: embeddings for text/categorical values
#[derive(Debug, Archive, Serialize, Deserialize)]
pub struct Database {
    // --- Schema ---
    pub tables: Vec<Table>,
    pub columns: Vec<Column>,

    // --- Data ---
    pub rows: Vec<Row>,

    // --- Graph (FK relationships) ---
    pub fk_edges: Vec<ForeignKeyEdge>,
    /// Outgoing edges from each row: row_idx -> [edge indices in fk_edges]
    pub outgoing_edges: Vec<Vec<usize>>,
    /// Incoming edges to each row: row_idx -> [edge indices in fk_edges]
    pub incoming_edges: Vec<Vec<usize>>,

    // --- Text Embedding Vocabulary ---
    /// Maps text values to their embedding index.
    pub text_lookup: HashMap<String, EmbeddingIdx>,
    /// Embeddings for each interned text value.
    pub text_embeddings: Vec<Vec<f16>>,

    // --- Global Timestamp Statistics ---
    pub timestamp_mean: Option<f64>,
    pub timestamp_std: Option<f64>,

    // --- Primary Key Index (for FK resolution) ---
    /// Maps (table_idx, pk_value) -> row_idx for integer PKs.
    pub pk_index: HashMap<(TableIdx, i64), RowIdx>,
}

impl Database {
    /// Create a new empty database.
    pub fn new() -> Self {
        Self {
            tables: Vec::new(),
            columns: Vec::new(),
            rows: Vec::new(),
            fk_edges: Vec::new(),
            outgoing_edges: Vec::new(),
            incoming_edges: Vec::new(),
            text_lookup: HashMap::new(),
            text_embeddings: Vec::new(),
            timestamp_mean: None,
            timestamp_std: None,
            pk_index: HashMap::new(),
        }
    }

    // --- Schema Accessors ---

    pub fn num_tables(&self) -> usize {
        self.tables.len()
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    pub fn num_edges(&self) -> usize {
        self.fk_edges.len()
    }

    pub fn vocab_size(&self) -> usize {
        self.text_embeddings.len()
    }

    pub fn table(&self, idx: TableIdx) -> &Table {
        &self.tables[idx.0]
    }

    pub fn column(&self, idx: ColumnIdx) -> &Column {
        &self.columns[idx.0]
    }

    pub fn row(&self, idx: RowIdx) -> &Row {
        &self.rows[idx.0]
    }

    pub fn table_name(&self, idx: TableIdx) -> &str {
        &self.tables[idx.0].name
    }

    pub fn column_name(&self, idx: ColumnIdx) -> &str {
        &self.columns[idx.0].name
    }

    /// Get the columns for a specific table.
    pub fn table_columns(&self, table_idx: TableIdx) -> &[Column] {
        let table = &self.tables[table_idx.0];
        &self.columns[table.column_range.0 .0..table.column_range.1 .0]
    }

    // --- Text Interning ---

    /// Get or insert a text value, computing its embedding if new.
    pub fn intern_text<F>(&mut self, text: &str, embed_fn: F) -> EmbeddingIdx
    where
        F: FnOnce(&str) -> Vec<f16>,
    {
        if let Some(&idx) = self.text_lookup.get(text) {
            return idx;
        }

        let idx = EmbeddingIdx(self.text_embeddings.len());
        self.text_lookup.insert(text.to_string(), idx);
        self.text_embeddings.push(embed_fn(text));
        idx
    }

    /// Reserve a slot for a text embedding (to be filled later during batch processing).
    pub fn reserve_text(&mut self, text: &str) -> EmbeddingIdx {
        if let Some(&idx) = self.text_lookup.get(text) {
            return idx;
        }

        let idx = EmbeddingIdx(self.text_embeddings.len());
        self.text_lookup.insert(text.to_string(), idx);
        self.text_embeddings.push(Vec::new()); // Placeholder
        idx
    }

    /// Set the embedding for a previously reserved text index.
    pub fn set_text_embedding(&mut self, idx: EmbeddingIdx, embedding: Vec<f16>) {
        if idx.0 < self.text_embeddings.len() {
            self.text_embeddings[idx.0] = embedding;
        }
    }

    // --- Graph Operations ---

    /// Build adjacency lists from fk_edges.
    pub fn build_adjacency(&mut self) {
        self.outgoing_edges = vec![Vec::new(); self.rows.len()];
        self.incoming_edges = vec![Vec::new(); self.rows.len()];

        for (edge_idx, edge) in self.fk_edges.iter().enumerate() {
            self.outgoing_edges[edge.from_row.0].push(edge_idx);
            self.incoming_edges[edge.to_row.0].push(edge_idx);
        }
    }

    /// Ensure adjacency lists exist and match the current row count.
    pub fn ensure_adjacency(&mut self) {
        if self.outgoing_edges.len() != self.rows.len() {
            self.build_adjacency();
        }
    }

    // --- Serialization ---

    /// Save the database to a file using rkyv.
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
        writer.flush()
    }

    /// Load a database from an rkyv file.
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
}

impl Default for Database {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_type_from_str() {
        assert_eq!(SemanticType::from_str("numerical"), Some(SemanticType::Numerical));
        assert_eq!(SemanticType::from_str("Categorical"), Some(SemanticType::Categorical));
        assert_eq!(SemanticType::from_str("TIMESTAMP"), Some(SemanticType::Timestamp));
        assert_eq!(SemanticType::from_str("text"), Some(SemanticType::Text));
        assert_eq!(SemanticType::from_str("invalid"), None);
    }

    #[test]
    fn test_timestamp_encoding() {
        // Test epoch 0 (Jan 1, 1970 00:00:00 UTC - Thursday)
        let cell = CellValue::from_epoch_seconds(0.0, 0.0, 1.0);
        if let CellValue::Timestamp(arr) = cell {
            // minute_of_hour = 0, so sin=0, cos=1
            assert!((arr[0] - 0.0).abs() < 1e-5);
            assert!((arr[1] - 1.0).abs() < 1e-5);
            // z-scored epoch should be 0
            assert!((arr[10] - 0.0).abs() < 1e-5);
        } else {
            panic!("Expected Timestamp variant");
        }
    }

    #[test]
    fn test_table_contains() {
        let table = Table {
            name: "test".to_string(),
            idx: TableIdx(0),
            column_range: (ColumnIdx(5), ColumnIdx(10)),
            row_range: (RowIdx(100), RowIdx(200)),
            primary_key_column: None,
            time_column: None,
        };

        assert!(table.contains_column(ColumnIdx(5)));
        assert!(table.contains_column(ColumnIdx(9)));
        assert!(!table.contains_column(ColumnIdx(10)));
        assert!(!table.contains_column(ColumnIdx(4)));

        assert!(table.contains_row(RowIdx(100)));
        assert!(table.contains_row(RowIdx(199)));
        assert!(!table.contains_row(RowIdx(200)));
        assert!(!table.contains_row(RowIdx(99)));
    }

    #[test]
    fn test_database_text_interning() {
        let mut db = Database::new();

        let idx1 = db.intern_text("hello", |_| vec![f16::from_f32(1.0)]);
        let idx2 = db.intern_text("world", |_| vec![f16::from_f32(2.0)]);
        let idx3 = db.intern_text("hello", |_| vec![f16::from_f32(3.0)]); // Should reuse idx1

        assert_eq!(idx1.0, 0);
        assert_eq!(idx2.0, 1);
        assert_eq!(idx3.0, 0); // Same as idx1
        assert_eq!(db.vocab_size(), 2);
    }

    #[test]
    fn test_cell_value_is_null() {
        assert!(CellValue::Null.is_null());
        assert!(!CellValue::Numerical(1.0).is_null());
        assert!(!CellValue::Categorical(EmbeddingIdx(0)).is_null());
    }
}
