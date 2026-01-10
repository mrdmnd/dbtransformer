//! Types for representing a relational database as a graph structure.
//!
//! This module provides:
//! - Metadata types for loading database schema from JSON
//! - Schema types (Table, Column) with semantic type information
//! - Normalized cell values ready for ML model consumption
//! - CSR graph representation via foreign key edges
//! - Flat, contiguous embedding storage
//!
//! Design principles:
//! - Preprocessing-only data (pk_index, text_lookup) lives in PreprocessingContext
//! - Runtime data uses cache-friendly flat arrays and CSR adjacency
//! - All indices are u32 to save memory (sufficient for <4B rows/edges)

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use half::f16;
use rkyv::{Archive, Deserialize, Serialize};
use serde::Deserialize as SerdeDeserialize;

// ============================================================================
// Index Types (u32 for memory efficiency)
// ============================================================================

/// Global table index in the database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, Hash, PartialEq, Eq))]
pub struct TableIdx(pub u32);

/// Global column index (unique across all tables) in the database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, Hash, PartialEq, Eq))]
pub struct ColumnIdx(pub u32);

/// Global row index (unique across all tables) in the database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, Hash, PartialEq, Eq))]
pub struct RowIdx(pub u32);

/// Index into the interned text embedding vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, Hash, PartialEq, Eq))]
pub struct EmbeddingIdx(pub u32);

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
    /// The column that serves as the primary key for this table - may not be present.
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
/// NOTE: The numeric values MUST match the Python SemanticType enum in model.py!
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[repr(u8)]
pub enum SemanticType {
    /// Numeric values where ordering is meaningful (price, age, quantity).
    /// Normalized via z-score: (value - mean) / std
    Numerical = 0,

    /// Discrete values from a finite set (status, type, category).
    /// Embedded as "<column_name> is <value>" text.
    Categorical = 1,

    /// Free-form text with semantic meaning (description, comment).
    /// Embedded via a frozen text embedding model.
    Text = 2,

    /// Date/time values with cyclical components.
    /// Encoded as sin/cos pairs for minute of hour, hour of day, day of week, day of year, month of year, plus z-scored epoch.
    Timestamp = 3,
}

impl SemanticType {
    /// Parse a semantic type from a string (for stype_overrides).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "numerical" => Some(Self::Numerical),
            "categorical" => Some(Self::Categorical),
            "text" => Some(Self::Text),
            "timestamp" => Some(Self::Timestamp),
            _ => None,
        }
    }
}

// ============================================================================
// Schema: Column
// ============================================================================

/// Column metadata and normalization parameters.
/// Note: description_embedding is stored separately in Database::column_embeddings
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

    /// For Numerical columns: z-score normalization parameters.
    pub norm_mean: Option<f32>,
    pub norm_std: Option<f32>,
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
        (self.column_range.1.0 - self.column_range.0.0) as usize
    }

    /// Number of rows in this table.
    #[inline]
    pub fn num_rows(&self) -> usize {
        (self.row_range.1.0 - self.row_range.0.0) as usize
    }

    /// Check if a column index belongs to this table.
    #[inline]
    pub fn contains_column(&self, col: ColumnIdx) -> bool {
        col.0 >= self.column_range.0.0 && col.0 < self.column_range.1.0
    }

    /// Check if a row index belongs to this table.
    #[inline]
    pub fn contains_row(&self, row: RowIdx) -> bool {
        row.0 >= self.row_range.0.0 && row.0 < self.row_range.1.0
    }
}

// ============================================================================
// Packed Cell Values (4 bytes per cell)
// ============================================================================

/// A packed cell value - interpretation depends on the column's SemanticType.
///
/// This representation saves 50% memory compared to an enum (4 bytes vs 8 bytes):
/// - Numerical: `f32::from_bits(value)` gives the z-scored scalar
/// - Categorical: `EmbeddingIdx(value)` gives the embedding index
/// - Timestamp: `f32::from_bits(value)` gives (fractional) epoch seconds
/// - Text: `EmbeddingIdx(value)` gives the embedding index
/// - Null: represented by `PACKED_NULL` sentinel
pub type PackedCell = u32;

/// Sentinel value for NULL cells.
/// This is a quiet NaN bit pattern (0x7FC00001) which:
/// - Won't appear naturally from f32::to_bits() on valid floats
/// - Is an absurdly large embedding index (~2.1 billion)
pub const PACKED_NULL: PackedCell = 0x7FC0_0001;

/// Pack a z-scored numerical value.
#[inline]
pub fn pack_numerical(v: f32) -> PackedCell {
    v.to_bits()
}

/// Pack epoch seconds for a timestamp.
#[inline]
pub fn pack_timestamp(epoch_secs: f32) -> PackedCell {
    epoch_secs.to_bits()
}

/// Pack an embedding index (for categorical or text).
#[inline]
pub fn pack_embedding_idx(idx: EmbeddingIdx) -> PackedCell {
    idx.0
}

/// Pack a null value.
#[inline]
pub fn pack_null() -> PackedCell {
    PACKED_NULL
}

/// Check if a packed cell is null.
#[inline]
pub fn is_packed_null(cell: PackedCell) -> bool {
    cell == PACKED_NULL
}

// ============================================================================
// CSR Graph (Compressed Sparse Row)
// ============================================================================

/// CSR (Compressed Sparse Row) representation for graph adjacency.
/// Much more cache-friendly than Vec<Vec<u32>>.
#[derive(Debug, Clone, Default, Archive, Serialize, Deserialize)]
pub struct CsrGraph {
    /// Row pointers: row_ptr[i] is the start index in col_idx for node i.
    /// Length = num_nodes + 1. row_ptr[num_nodes] = num_edges.
    pub row_ptr: Vec<u32>,
    /// Column indices (neighbor node IDs) for each edge.
    /// Length = num_edges.
    pub col_idx: Vec<u32>,
}

impl CsrGraph {
    /// Create a new empty CSR graph.
    pub fn new() -> Self {
        Self {
            row_ptr: vec![0],
            col_idx: Vec::new(),
        }
    }

    /// Create a CSR graph with the given capacity.
    pub fn with_capacity(num_nodes: usize, num_edges: usize) -> Self {
        let mut row_ptr = Vec::with_capacity(num_nodes + 1);
        row_ptr.push(0);
        Self {
            row_ptr,
            col_idx: Vec::with_capacity(num_edges),
        }
    }

    /// Number of nodes in the graph.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.row_ptr.len().saturating_sub(1)
    }

    /// Number of edges in the graph.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.col_idx.len()
    }

    /// Get the neighbors of a node.
    #[inline]
    pub fn neighbors(&self, node: u32) -> &[u32] {
        let start = self.row_ptr[node as usize] as usize;
        let end = self.row_ptr[node as usize + 1] as usize;
        &self.col_idx[start..end]
    }

    /// Get the degree (number of neighbors) of a node.
    #[inline]
    pub fn degree(&self, node: u32) -> usize {
        let start = self.row_ptr[node as usize];
        let end = self.row_ptr[node as usize + 1];
        (end - start) as usize
    }

    /// Build CSR from an edge list. Edges must be sorted by source node.
    pub fn from_sorted_edges(num_nodes: usize, edges: &[(u32, u32)]) -> Self {
        let mut row_ptr = Vec::with_capacity(num_nodes + 1);
        let mut col_idx = Vec::with_capacity(edges.len());

        let mut current_node = 0u32;
        row_ptr.push(0);

        for &(src, dst) in edges {
            // Fill in empty rows
            while current_node < src {
                row_ptr.push(col_idx.len() as u32);
                current_node += 1;
            }
            col_idx.push(dst);
        }

        // Fill remaining rows
        while row_ptr.len() <= num_nodes {
            row_ptr.push(col_idx.len() as u32);
        }

        Self { row_ptr, col_idx }
    }

    /// Build CSR from an unsorted edge list.
    pub fn from_edges(num_nodes: usize, edges: &mut [(u32, u32)]) -> Self {
        edges.sort_unstable_by_key(|e| e.0);
        Self::from_sorted_edges(num_nodes, edges)
    }
}

// ============================================================================
// Preprocessing Context (thrown away after preprocessing)
// ============================================================================

/// Context used only during preprocessing. Not serialized.
/// Contains lookup structures needed to build the graph.
#[derive(Debug, Default)]
pub struct PreprocessingContext {
    /// Maps (table_idx, pk_value) -> row_idx for integer PKs.
    /// Used to resolve FK references during edge building.
    pub pk_index: HashMap<(TableIdx, i64), RowIdx>,

    /// Maps text values to their embedding index.
    /// Used to deduplicate text during cell processing.
    pub text_lookup: HashMap<String, EmbeddingIdx>,

    /// Pending texts that need embedding (in order of EmbeddingIdx).
    pub pending_texts: Vec<String>,
}

impl PreprocessingContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a text value, returning its embedding index.
    /// If the text is new, it's added to pending_texts for later embedding.
    pub fn intern_text(&mut self, text: &str) -> EmbeddingIdx {
        if let Some(&idx) = self.text_lookup.get(text) {
            return idx;
        }

        let idx = EmbeddingIdx(self.pending_texts.len() as u32);
        self.text_lookup.insert(text.to_string(), idx);
        self.pending_texts.push(text.to_string());
        idx
    }

    /// Get the number of unique texts interned.
    pub fn vocab_size(&self) -> usize {
        self.pending_texts.len()
    }

    /// Look up the row for a primary key value.
    pub fn lookup_pk(&self, table_idx: TableIdx, pk_value: i64) -> Option<RowIdx> {
        self.pk_index.get(&(table_idx, pk_value)).copied()
    }

    /// Register a primary key -> row mapping.
    pub fn register_pk(&mut self, table_idx: TableIdx, pk_value: i64, row_idx: RowIdx) {
        self.pk_index.insert((table_idx, pk_value), row_idx);
    }
}

// ============================================================================
// Database (the final ML-ready structure)
// ============================================================================

/// A complete preprocessed database ready for sampling.
///
/// Design:
/// - Schema: tables and columns with metadata (small, kept for introspection)
/// - Cell data: flat array of packed u32 values with row offsets
/// - Graph: CSR adjacency for outgoing/incoming FK edges
/// - Embeddings: flat contiguous buffer indexed by EmbeddingIdx
/// - Column embeddings: flat contiguous buffer indexed by ColumnIdx
#[derive(Debug, Archive, Serialize, Deserialize)]
pub struct Database {
    // --- Schema ---
    pub tables: Vec<Table>,
    pub columns: Vec<Column>,

    // --- Cell Data (flat with row offsets) ---
    /// All cell values as packed u32s, stored row-by-row.
    /// For row i: values are at cell_values[row_offsets[i]..row_offsets[i+1]]
    /// Interpretation depends on column's SemanticType (see `unpack_cell`).
    pub cell_values: Vec<PackedCell>,
    /// row_offsets[i] = start index in cell_values for row i.
    /// Length = num_rows + 1, where row_offsets[num_rows] = cell_values.len()
    pub row_offsets: Vec<u32>,

    // --- Graph (CSR adjacency) ---
    /// Outgoing edges: from row -> to rows
    pub outgoing: CsrGraph,
    /// Incoming edges: to row <- from rows
    pub incoming: CsrGraph,

    // --- Text Embeddings (flat, contiguous) ---
    /// Flat buffer of text embeddings: shape (vocab_size, embed_dim).
    /// Index with: embeddings[idx.0 * embed_dim..(idx.0 + 1) * embed_dim]
    pub embeddings: Vec<f16>,
    /// Embedding dimension (e.g., 768 for BGE).
    pub embed_dim: u32,

    // --- Column Embeddings (flat, contiguous) ---
    /// Flat buffer of column description embeddings: shape (num_cols, embed_dim).
    /// Index with: column_embeddings[col_idx.0 * embed_dim..(col_idx.0 + 1) * embed_dim]
    pub column_embeddings: Vec<f16>,

    // --- Global Timestamp Statistics ---
    pub timestamp_mean: Option<f64>,
    pub timestamp_std: Option<f64>,
}

impl Database {
    /// Create a new empty database.
    pub fn new() -> Self {
        Self {
            tables: Vec::new(),
            columns: Vec::new(),
            cell_values: Vec::new(),
            row_offsets: vec![0], // Always starts with 0
            outgoing: CsrGraph::new(),
            incoming: CsrGraph::new(),
            embeddings: Vec::new(),
            embed_dim: 0,
            column_embeddings: Vec::new(),
            timestamp_mean: None,
            timestamp_std: None,
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
        self.row_offsets.len().saturating_sub(1)
    }

    pub fn num_edges(&self) -> usize {
        self.outgoing.num_edges()
    }

    pub fn vocab_size(&self) -> usize {
        if self.embed_dim == 0 {
            0
        } else {
            self.embeddings.len() / self.embed_dim as usize
        }
    }

    pub fn table(&self, idx: TableIdx) -> &Table {
        &self.tables[idx.0 as usize]
    }

    pub fn column(&self, idx: ColumnIdx) -> &Column {
        &self.columns[idx.0 as usize]
    }

    pub fn table_name(&self, idx: TableIdx) -> &str {
        &self.tables[idx.0 as usize].name
    }

    pub fn column_name(&self, idx: ColumnIdx) -> &str {
        &self.columns[idx.0 as usize].name
    }

    /// Get the columns for a specific table.
    pub fn table_columns(&self, table_idx: TableIdx) -> &[Column] {
        let table = &self.tables[table_idx.0 as usize];
        &self.columns[table.column_range.0.0 as usize..table.column_range.1.0 as usize]
    }

    /// Get the packed cells for a specific row.
    #[inline]
    pub fn row_cells(&self, row_idx: RowIdx) -> &[PackedCell] {
        let start = self.row_offsets[row_idx.0 as usize] as usize;
        let end = self.row_offsets[row_idx.0 as usize + 1] as usize;
        &self.cell_values[start..end]
    }

    /// Get which table a row belongs to.
    /// Uses binary search for O(log n) instead of O(n) linear search.
    #[inline]
    pub fn row_table(&self, row_idx: RowIdx) -> TableIdx {
        let row = row_idx.0;
        // Binary search: find the table whose row_range contains this row
        let idx = self
            .tables
            .binary_search_by(|table| {
                if row < table.row_range.0.0 {
                    std::cmp::Ordering::Greater
                } else if row >= table.row_range.1.0 {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .expect("Row not found in any table");
        self.tables[idx].idx
    }

    // --- Embedding Access ---

    /// Get the embedding for a text/categorical value.
    #[inline]
    pub fn get_embedding(&self, idx: EmbeddingIdx) -> &[f16] {
        let start = idx.0 as usize * self.embed_dim as usize;
        let end = start + self.embed_dim as usize;
        &self.embeddings[start..end]
    }

    /// Get the embedding for a column description.
    #[inline]
    pub fn get_column_embedding(&self, idx: ColumnIdx) -> &[f16] {
        let start = idx.0 as usize * self.embed_dim as usize;
        let end = start + self.embed_dim as usize;
        &self.column_embeddings[start..end]
    }

    /// Set the embedding for a text/categorical value.
    pub fn set_embedding(&mut self, idx: EmbeddingIdx, embedding: &[f16]) {
        debug_assert_eq!(embedding.len(), self.embed_dim as usize);
        let start = idx.0 as usize * self.embed_dim as usize;
        self.embeddings[start..start + embedding.len()].copy_from_slice(embedding);
    }

    /// Set the embedding for a column description.
    pub fn set_column_embedding(&mut self, idx: ColumnIdx, embedding: &[f16]) {
        debug_assert_eq!(embedding.len(), self.embed_dim as usize);
        let start = idx.0 as usize * self.embed_dim as usize;
        self.column_embeddings[start..start + embedding.len()].copy_from_slice(embedding);
    }

    // --- Graph Operations ---

    /// Get outgoing neighbors (rows this row points to via FK).
    #[inline]
    pub fn outgoing_neighbors(&self, row: RowIdx) -> &[u32] {
        self.outgoing.neighbors(row.0)
    }

    /// Get incoming neighbors (rows that point to this row via FK).
    #[inline]
    pub fn incoming_neighbors(&self, row: RowIdx) -> &[u32] {
        self.incoming.neighbors(row.0)
    }

    /// Build CSR graphs from edge list.
    /// Edges: Vec<(from_row, to_row)>
    pub fn build_csr_from_edges(&mut self, mut edges: Vec<(u32, u32)>) {
        let num_nodes = self.num_rows();

        // Build outgoing CSR
        edges.sort_unstable_by_key(|e| e.0);
        self.outgoing = CsrGraph::from_sorted_edges(num_nodes, &edges);

        // Build incoming CSR (reverse edges)
        let mut incoming_edges: Vec<(u32, u32)> = edges.iter().map(|&(f, t)| (t, f)).collect();
        incoming_edges.sort_unstable_by_key(|e| e.0);
        self.incoming = CsrGraph::from_sorted_edges(num_nodes, &incoming_edges);
    }

    // --- Cell Storage ---

    /// Reserve capacity for the expected number of cells.
    pub fn reserve_cells(&mut self, total_cells: usize, total_rows: usize) {
        self.cell_values.reserve(total_cells);
        self.row_offsets.reserve(total_rows + 1);
    }

    /// Append a row of packed cells. Must be called in row order (row 0, 1, 2, ...).
    #[inline]
    pub fn push_row(&mut self, cells: &[PackedCell]) {
        self.cell_values.extend_from_slice(cells);
        self.row_offsets.push(self.cell_values.len() as u32);
    }

    // --- Initialization Helpers ---

    /// Initialize embedding storage with the given dimension and vocab size.
    pub fn init_embeddings(&mut self, embed_dim: u32, vocab_size: usize) {
        self.embed_dim = embed_dim;
        self.embeddings = vec![f16::ZERO; vocab_size * embed_dim as usize];
    }

    /// Initialize column embedding storage.
    pub fn init_column_embeddings(&mut self, embed_dim: u32) {
        self.embed_dim = embed_dim;
        self.column_embeddings = vec![f16::ZERO; self.columns.len() * embed_dim as usize];
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

        rkyv::from_bytes::<Self, rkyv::rancor::Error>(&bytes).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Deserialization error: {e}"),
            )
        })
    }
}

impl Default for Database {
    fn default() -> Self {
        Self::new()
    }
}

impl Database {
    /// Print the memory size of each field in the Database.
    pub fn print_field_sizes(&self) {
        use std::mem::size_of_val;
        use tracing::info;

        let tables_bytes = size_of_val(&*self.tables)
            + self.tables.iter().map(|t| t.name.capacity()).sum::<usize>();
        let columns_bytes = size_of_val(&*self.columns)
            + self
                .columns
                .iter()
                .map(|c| c.name.capacity())
                .sum::<usize>();
        let cell_values_bytes = size_of_val(&*self.cell_values);
        let row_offsets_bytes = size_of_val(&*self.row_offsets);
        let outgoing_row_ptr_bytes = size_of_val(&*self.outgoing.row_ptr);
        let outgoing_col_idx_bytes = size_of_val(&*self.outgoing.col_idx);
        let incoming_row_ptr_bytes = size_of_val(&*self.incoming.row_ptr);
        let incoming_col_idx_bytes = size_of_val(&*self.incoming.col_idx);
        let embeddings_bytes = size_of_val(&*self.embeddings);
        let column_embeddings_bytes = size_of_val(&*self.column_embeddings);

        let total_bytes = tables_bytes
            + columns_bytes
            + cell_values_bytes
            + row_offsets_bytes
            + outgoing_row_ptr_bytes
            + outgoing_col_idx_bytes
            + incoming_row_ptr_bytes
            + incoming_col_idx_bytes
            + embeddings_bytes
            + column_embeddings_bytes;

        info!("=== Database Field Sizes ===");
        info!(bytes = tables_bytes, count = self.tables.len(), "tables");
        info!(bytes = columns_bytes, count = self.columns.len(), "columns");
        info!(
            bytes = cell_values_bytes,
            count = self.cell_values.len(),
            "cell_values"
        );
        info!(
            bytes = row_offsets_bytes,
            count = self.row_offsets.len(),
            "row_offsets"
        );
        info!(
            bytes = outgoing_row_ptr_bytes,
            count = self.outgoing.row_ptr.len(),
            "outgoing.row_ptr"
        );
        info!(
            bytes = outgoing_col_idx_bytes,
            edges = self.outgoing.col_idx.len(),
            "outgoing.col_idx"
        );
        info!(
            bytes = incoming_row_ptr_bytes,
            count = self.incoming.row_ptr.len(),
            "incoming.row_ptr"
        );
        info!(
            bytes = incoming_col_idx_bytes,
            edges = self.incoming.col_idx.len(),
            "incoming.col_idx"
        );
        info!(
            bytes = embeddings_bytes,
            f16_values = self.embeddings.len(),
            vocab = self.vocab_size(),
            "embeddings"
        );
        info!(
            bytes = column_embeddings_bytes,
            f16_values = self.column_embeddings.len(),
            "column_embeddings"
        );
        info!(
            total_bytes,
            total_mb = format!("{:.2}", total_bytes as f64 / 1_048_576.0),
            "TOTAL"
        );
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
        assert_eq!(
            SemanticType::from_str("numerical"),
            Some(SemanticType::Numerical)
        );
        assert_eq!(
            SemanticType::from_str("Categorical"),
            Some(SemanticType::Categorical)
        );
        assert_eq!(
            SemanticType::from_str("TIMESTAMP"),
            Some(SemanticType::Timestamp)
        );
        assert_eq!(SemanticType::from_str("text"), Some(SemanticType::Text));
        assert_eq!(SemanticType::from_str("invalid"), None);
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
    fn test_csr_graph() {
        // Build a simple graph: 0->1, 0->2, 1->2, 2->0
        let mut edges = vec![(0, 1), (0, 2), (1, 2), (2, 0)];
        let csr = CsrGraph::from_edges(3, &mut edges);

        assert_eq!(csr.num_nodes(), 3);
        assert_eq!(csr.num_edges(), 4);

        assert_eq!(csr.neighbors(0), &[1, 2]);
        assert_eq!(csr.neighbors(1), &[2]);
        assert_eq!(csr.neighbors(2), &[0]);

        assert_eq!(csr.degree(0), 2);
        assert_eq!(csr.degree(1), 1);
        assert_eq!(csr.degree(2), 1);
    }

    #[test]
    fn test_preprocessing_context() {
        let mut ctx = PreprocessingContext::new();

        let idx1 = ctx.intern_text("hello");
        let idx2 = ctx.intern_text("world");
        let idx3 = ctx.intern_text("hello"); // Should reuse idx1

        assert_eq!(idx1.0, 0);
        assert_eq!(idx2.0, 1);
        assert_eq!(idx3.0, 0); // Same as idx1
        assert_eq!(ctx.vocab_size(), 2);

        ctx.register_pk(TableIdx(0), 42, RowIdx(100));
        assert_eq!(ctx.lookup_pk(TableIdx(0), 42), Some(RowIdx(100)));
        assert_eq!(ctx.lookup_pk(TableIdx(0), 99), None);
    }

    #[test]
    fn test_packed_cell_roundtrip() {
        // Test numerical roundtrip
        let val = 3.14159f32;
        let packed = pack_numerical(val);
        assert_eq!(f32::from_bits(packed), val);

        // Test timestamp roundtrip
        let epoch = 1704067200.0f32; // Jan 1, 2024
        let packed = pack_timestamp(epoch);
        assert_eq!(f32::from_bits(packed), epoch);

        // Test embedding index roundtrip
        let idx = EmbeddingIdx(12345);
        let packed = pack_embedding_idx(idx);
        assert_eq!(packed, 12345);

        // Test null
        let packed = pack_null();
        assert!(is_packed_null(packed));
        assert!(!is_packed_null(0));
        assert!(!is_packed_null(pack_numerical(0.0)));
    }

    #[test]
    fn test_database_embeddings() {
        let mut db = Database::new();
        db.columns.push(Column {
            name: "test".to_string(),
            idx: ColumnIdx(0),
            table_idx: TableIdx(0),
            stype: SemanticType::Numerical,
            is_primary_key: false,
            fk_target_column: None,
            norm_mean: None,
            norm_std: None,
        });

        // Initialize embeddings
        db.init_embeddings(4, 2); // 4-dim embeddings, vocab size 2
        db.init_column_embeddings(4);

        // Set embeddings
        let emb1 = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let emb2 = [
            f16::from_f32(5.0),
            f16::from_f32(6.0),
            f16::from_f32(7.0),
            f16::from_f32(8.0),
        ];

        db.set_embedding(EmbeddingIdx(0), &emb1);
        db.set_embedding(EmbeddingIdx(1), &emb2);
        db.set_column_embedding(ColumnIdx(0), &emb1);

        // Verify
        assert_eq!(db.get_embedding(EmbeddingIdx(0)), &emb1);
        assert_eq!(db.get_embedding(EmbeddingIdx(1)), &emb2);
        assert_eq!(db.get_column_embedding(ColumnIdx(0)), &emb1);
        assert_eq!(db.vocab_size(), 2);
    }
}
