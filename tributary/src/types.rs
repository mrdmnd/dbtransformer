//! Types for representing a relational database as a graph structure.
//!
//! ## File Layout
//!
//! A preprocessed database is stored as a directory with separate files:
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
//! ## Design Principles
//!
//! - Each component is independently mmap-able for multi-process sharing
//! - Schema is small and always loaded into memory
//! - Graph, cells, embeddings are mmap'd for zero-copy access
//! - All indices are u32 to save memory (sufficient for <4B rows/edges)

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use half::f16;
use memmap2::Mmap;
use rkyv::{Archive, Deserialize, Serialize};
use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};

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
// Metadata Types (for loading from metadata.json in raw database)
// ============================================================================

/// Per-table metadata loaded from metadata.json.
#[derive(Debug, Clone, SerdeDeserialize)]
pub struct TableMetadata {
    /// The column that serves as the primary key for this table.
    pub primary_key_column: Option<String>,

    /// Maps FK column names to the table they reference.
    #[serde(default)]
    pub foreign_key_column_to_primary_key_table: HashMap<String, String>,

    /// Optional timestamp column indicating when each row's data became valid.
    pub time_column: Option<String>,

    /// Override the auto-detected semantic type for specific columns.
    #[serde(default)]
    pub stype_overrides: HashMap<String, String>,

    /// Columns to exclude from processing entirely.
    #[serde(default)]
    pub ignored_columns: Vec<String>,

    /// Columns that are valid prediction targets (can be masked during training).
    /// If empty or not specified, NO columns from this table are prediction targets.
    /// This prevents data leakage from redundant/denormalized columns.
    #[serde(default)]
    pub prediction_targets: Vec<String>,
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
    /// Numeric values normalized via z-score.
    Numerical = 0,
    /// Discrete values embedded as "column_name is value".
    Categorical = 1,
    /// Free-form text embedded via text model.
    Text = 2,
    /// Date/time values with cyclical encoding.
    Timestamp = 3,
}

impl SemanticType {
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
// Schema Types
// ============================================================================

/// Column metadata and normalization parameters.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    pub idx: ColumnIdx,
    pub table_idx: TableIdx,
    pub stype: SemanticType,
    pub is_primary_key: bool,
    pub fk_target_column: Option<ColumnIdx>,
    pub norm_mean: Option<f32>,
    pub norm_std: Option<f32>,
    /// Whether this column can be masked as a prediction target.
    /// Only columns explicitly listed in metadata.prediction_targets are true.
    pub is_prediction_target: bool,
}

/// Table metadata with column and row ranges.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct Table {
    pub name: String,
    pub idx: TableIdx,
    /// Range of column indices [start, end).
    pub column_range: (ColumnIdx, ColumnIdx),
    /// Range of row indices [start, end).
    pub row_range: (RowIdx, RowIdx),
    /// Feature columns stored as cells (excludes PK/FK).
    pub feature_columns: Vec<ColumnIdx>,
    pub primary_key_column: Option<ColumnIdx>,
    pub time_column: Option<ColumnIdx>,
}

impl Table {
    #[inline]
    pub fn num_columns(&self) -> usize {
        (self.column_range.1.0 - self.column_range.0.0) as usize
    }

    #[inline]
    pub fn num_feature_columns(&self) -> usize {
        self.feature_columns.len()
    }

    #[inline]
    pub fn num_rows(&self) -> usize {
        (self.row_range.1.0 - self.row_range.0.0) as usize
    }

    #[inline]
    pub fn contains_column(&self, col: ColumnIdx) -> bool {
        col.0 >= self.column_range.0.0 && col.0 < self.column_range.1.0
    }

    #[inline]
    pub fn contains_row(&self, row: RowIdx) -> bool {
        row.0 >= self.row_range.0.0 && row.0 < self.row_range.1.0
    }

    #[inline]
    pub fn cell_column(&self, cell_pos: usize) -> ColumnIdx {
        self.feature_columns[cell_pos]
    }
}

// ============================================================================
// Packed Cell Values
// ============================================================================

/// A packed cell value - interpretation depends on column's SemanticType.
/// - Numerical: f32::from_bits(value) gives the z-scored scalar
/// - Categorical: EmbeddingIdx(value) gives the embedding index
/// - Timestamp: f32::from_bits(value) gives epoch seconds
/// - Text: EmbeddingIdx(value) gives the embedding index
/// - Null: PACKED_NULL sentinel
pub type PackedCell = u32;

/// Sentinel value for NULL cells (quiet NaN bit pattern).
pub const PACKED_NULL: PackedCell = 0x7FC0_0001;

/// Sentinel for rows without a timestamp.
pub const NO_TIMESTAMP: i64 = i64::MIN;

#[inline]
pub fn pack_numerical(v: f32) -> PackedCell {
    v.to_bits()
}

#[inline]
pub fn pack_timestamp(epoch_secs: f32) -> PackedCell {
    epoch_secs.to_bits()
}

#[inline]
pub fn pack_embedding_idx(idx: EmbeddingIdx) -> PackedCell {
    idx.0
}

#[inline]
pub fn pack_null() -> PackedCell {
    PACKED_NULL
}

#[inline]
pub fn is_packed_null(cell: PackedCell) -> bool {
    cell == PACKED_NULL
}

// ============================================================================
// CSR Graph
// ============================================================================

/// Compressed Sparse Row representation for graph adjacency.
#[derive(Debug, Clone, Default, Archive, Serialize, Deserialize)]
pub struct CsrGraph {
    /// row_ptr[i] is the start index in col_idx for node i.
    /// Length = num_nodes + 1.
    pub row_ptr: Vec<u32>,
    /// Neighbor node IDs for each edge.
    pub col_idx: Vec<u32>,
}

impl CsrGraph {
    pub fn new() -> Self {
        Self {
            row_ptr: vec![0],
            col_idx: Vec::new(),
        }
    }

    pub fn with_capacity(num_nodes: usize, num_edges: usize) -> Self {
        let mut row_ptr = Vec::with_capacity(num_nodes + 1);
        row_ptr.push(0);
        Self {
            row_ptr,
            col_idx: Vec::with_capacity(num_edges),
        }
    }

    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.row_ptr.len().saturating_sub(1)
    }

    #[inline]
    pub fn num_edges(&self) -> usize {
        self.col_idx.len()
    }

    #[inline]
    pub fn neighbors(&self, node: u32) -> &[u32] {
        let start = self.row_ptr[node as usize] as usize;
        let end = self.row_ptr[node as usize + 1] as usize;
        &self.col_idx[start..end]
    }

    #[inline]
    pub fn degree(&self, node: u32) -> usize {
        let start = self.row_ptr[node as usize];
        let end = self.row_ptr[node as usize + 1];
        (end - start) as usize
    }

    pub fn from_sorted_edges(num_nodes: usize, edges: &[(u32, u32)]) -> Self {
        let mut row_ptr = Vec::with_capacity(num_nodes + 1);
        let mut col_idx = Vec::with_capacity(edges.len());

        let mut current_node = 0u32;
        row_ptr.push(0);

        for &(src, dst) in edges {
            while current_node < src {
                row_ptr.push(col_idx.len() as u32);
                current_node += 1;
            }
            col_idx.push(dst);
        }

        while row_ptr.len() <= num_nodes {
            row_ptr.push(col_idx.len() as u32);
        }

        Self { row_ptr, col_idx }
    }

    pub fn from_edges(num_nodes: usize, edges: &mut [(u32, u32)]) -> Self {
        edges.sort_unstable_by_key(|e| e.0);
        Self::from_sorted_edges(num_nodes, edges)
    }
}

// ============================================================================
// Preprocessing Context (thrown away after preprocessing)
// ============================================================================

#[derive(Debug, Default)]
pub struct PreprocessingContext {
    pub pk_index: HashMap<(TableIdx, i64), RowIdx>,
    pub text_lookup: HashMap<String, EmbeddingIdx>,
    pub pending_texts: Vec<String>,
}

impl PreprocessingContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern_text(&mut self, text: &str) -> EmbeddingIdx {
        if let Some(&idx) = self.text_lookup.get(text) {
            return idx;
        }
        let idx = EmbeddingIdx(self.pending_texts.len() as u32);
        self.text_lookup.insert(text.to_string(), idx);
        self.pending_texts.push(text.to_string());
        idx
    }

    pub fn vocab_size(&self) -> usize {
        self.pending_texts.len()
    }

    pub fn lookup_pk(&self, table_idx: TableIdx, pk_value: i64) -> Option<RowIdx> {
        self.pk_index.get(&(table_idx, pk_value)).copied()
    }

    pub fn register_pk(&mut self, table_idx: TableIdx, pk_value: i64, row_idx: RowIdx) {
        self.pk_index.insert((table_idx, pk_value), row_idx);
    }
}

// ============================================================================
// Split File Types
// ============================================================================

/// Schema file: tables, columns, column embeddings, global stats.
/// Small enough to always load into memory.
#[derive(Debug, Archive, Serialize, Deserialize)]
pub struct Schema {
    pub tables: Vec<Table>,
    pub columns: Vec<Column>,
    /// Flat buffer of column description embeddings: shape (num_cols, embed_dim).
    pub column_embeddings: Vec<f16>,
    pub embed_dim: u32,
    pub timestamp_mean: Option<f64>,
    pub timestamp_std: Option<f64>,
}

impl Schema {
    pub fn new() -> Self {
        Self {
            tables: Vec::new(),
            columns: Vec::new(),
            column_embeddings: Vec::new(),
            embed_dim: 0,
            timestamp_mean: None,
            timestamp_std: None,
        }
    }

    pub fn num_tables(&self) -> usize {
        self.tables.len()
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn table(&self, idx: TableIdx) -> &Table {
        &self.tables[idx.0 as usize]
    }

    pub fn column(&self, idx: ColumnIdx) -> &Column {
        &self.columns[idx.0 as usize]
    }

    pub fn init_column_embeddings(&mut self, embed_dim: u32) {
        self.embed_dim = embed_dim;
        self.column_embeddings = vec![f16::ZERO; self.columns.len() * embed_dim as usize];
    }

    pub fn set_column_embedding(&mut self, idx: ColumnIdx, embedding: &[f16]) {
        let start = idx.0 as usize * self.embed_dim as usize;
        self.column_embeddings[start..start + embedding.len()].copy_from_slice(embedding);
    }

    pub fn get_column_embedding(&self, idx: ColumnIdx) -> &[f16] {
        let start = idx.0 as usize * self.embed_dim as usize;
        let end = start + self.embed_dim as usize;
        &self.column_embeddings[start..end]
    }

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
}

impl Default for Schema {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph file: CSR adjacency for FK edges.
#[derive(Debug, Archive, Serialize, Deserialize)]
pub struct Graph {
    pub outgoing: CsrGraph,
    pub incoming: CsrGraph,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            outgoing: CsrGraph::new(),
            incoming: CsrGraph::new(),
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.outgoing.num_nodes()
    }

    pub fn num_edges(&self) -> usize {
        self.outgoing.num_edges()
    }

    #[inline]
    pub fn outgoing_neighbors(&self, row: RowIdx) -> &[u32] {
        self.outgoing.neighbors(row.0)
    }

    #[inline]
    pub fn incoming_neighbors(&self, row: RowIdx) -> &[u32] {
        self.incoming.neighbors(row.0)
    }

    pub fn build_from_edges(&mut self, num_nodes: usize, mut edges: Vec<(u32, u32)>) {
        edges.sort_unstable_by_key(|e| e.0);
        self.outgoing = CsrGraph::from_sorted_edges(num_nodes, &edges);

        let mut incoming_edges: Vec<(u32, u32)> = edges.iter().map(|&(f, t)| (t, f)).collect();
        incoming_edges.sort_unstable_by_key(|e| e.0);
        self.incoming = CsrGraph::from_sorted_edges(num_nodes, &incoming_edges);
    }

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
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// Cells file: packed cell values and row metadata.
#[derive(Debug, Archive, Serialize, Deserialize)]
pub struct Cells {
    /// All cell values as packed u32s, stored row-by-row.
    pub cell_values: Vec<PackedCell>,
    /// row_offsets[i] = start index in cell_values for row i.
    /// Length = num_rows + 1.
    pub row_offsets: Vec<u32>,
    /// Raw epoch seconds for each row (i64::MIN if no time_column).
    pub row_timestamps: Vec<i64>,
}

impl Cells {
    pub fn new() -> Self {
        Self {
            cell_values: Vec::new(),
            row_offsets: vec![0],
            row_timestamps: Vec::new(),
        }
    }

    pub fn num_rows(&self) -> usize {
        self.row_offsets.len().saturating_sub(1)
    }

    pub fn reserve(&mut self, total_cells: usize, total_rows: usize) {
        self.cell_values.reserve(total_cells);
        self.row_offsets.reserve(total_rows + 1);
        self.row_timestamps.reserve(total_rows);
    }

    #[inline]
    pub fn push_row(&mut self, cells: &[PackedCell], timestamp: i64) {
        self.cell_values.extend_from_slice(cells);
        self.row_offsets.push(self.cell_values.len() as u32);
        self.row_timestamps.push(timestamp);
    }

    #[inline]
    pub fn row_cells(&self, row_idx: RowIdx) -> &[PackedCell] {
        let start = self.row_offsets[row_idx.0 as usize] as usize;
        let end = self.row_offsets[row_idx.0 as usize + 1] as usize;
        &self.cell_values[start..end]
    }

    #[inline]
    pub fn row_timestamp(&self, row: RowIdx) -> Option<i64> {
        let ts = self.row_timestamps[row.0 as usize];
        if ts == NO_TIMESTAMP { None } else { Some(ts) }
    }

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
}

impl Default for Cells {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Manifest (JSON metadata)
// ============================================================================

#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize)]
pub struct ManifestStats {
    pub num_tables: usize,
    pub num_columns: usize,
    pub num_rows: usize,
    pub num_edges: usize,
    pub vocab_size: usize,
    pub embed_dim: usize,
}

#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize)]
pub struct Manifest {
    pub version: String,
    pub created: String,
    pub source_dir: String,
    pub stats: ManifestStats,
}

impl Manifest {
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to parse manifest.json: {e}"),
            )
        })
    }
}

// ============================================================================
// Memory-Mapped Database (combines all split files)
// ============================================================================

/// A memory-mapped database assembled from split files.
///
/// Each component is independently mmap'd for efficient multi-process access.
pub struct Database {
    /// Schema (loaded into memory, small)
    schema_mmap: Arc<Mmap>,
    /// Graph (mmap'd)
    graph_mmap: Arc<Mmap>,
    /// Cells (mmap'd)
    cells_mmap: Arc<Mmap>,
    /// Text embeddings (mmap'd)
    embeddings_mmap: Arc<Mmap>,
    /// Embedding dimension
    embed_dim: usize,
    /// Vocabulary size
    vocab_size: usize,
}

// SAFETY: All mmaps are read-only and immutable.
unsafe impl Send for Database {}
unsafe impl Sync for Database {}

impl Database {
    /// Load a database from a directory containing split files.
    pub fn load<P: AsRef<Path>>(dir: P) -> std::io::Result<Self> {
        let dir = dir.as_ref();

        // Load manifest to get metadata
        let _manifest = Manifest::load(dir.join("manifest.json"))?;

        // Memory-map all files
        let schema_file = File::open(dir.join("schema.rkyv"))?;
        let schema_mmap = unsafe { Mmap::map(&schema_file)? };

        let graph_file = File::open(dir.join("graph.rkyv"))?;
        let graph_mmap = unsafe { Mmap::map(&graph_file)? };

        let cells_file = File::open(dir.join("cells.rkyv"))?;
        let cells_mmap = unsafe { Mmap::map(&cells_file)? };

        let embeddings_file = File::open(dir.join("embeddings.bin"))?;
        let embeddings_mmap = unsafe { Mmap::map(&embeddings_file)? };

        // Parse embeddings header
        if embeddings_mmap.len() < 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Embeddings file too small",
            ));
        }
        let vocab_size = u32::from_le_bytes(embeddings_mmap[0..4].try_into().unwrap()) as usize;
        let embed_dim = u32::from_le_bytes(embeddings_mmap[4..8].try_into().unwrap()) as usize;

        // Validate
        let expected_size = 8 + vocab_size * embed_dim * 2;
        if embeddings_mmap.len() < expected_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Embeddings file truncated: expected {} bytes, got {}",
                    expected_size,
                    embeddings_mmap.len()
                ),
            ));
        }

        Ok(Self {
            schema_mmap: Arc::new(schema_mmap),
            graph_mmap: Arc::new(graph_mmap),
            cells_mmap: Arc::new(cells_mmap),
            embeddings_mmap: Arc::new(embeddings_mmap),
            embed_dim,
            vocab_size,
        })
    }

    // --- Archived accessors ---

    #[inline]
    fn archived_schema(&self) -> &ArchivedSchema {
        unsafe { rkyv::access_unchecked::<ArchivedSchema>(&self.schema_mmap) }
    }

    #[inline]
    fn archived_graph(&self) -> &ArchivedGraph {
        unsafe { rkyv::access_unchecked::<ArchivedGraph>(&self.graph_mmap) }
    }

    #[inline]
    fn archived_cells(&self) -> &ArchivedCells {
        unsafe { rkyv::access_unchecked::<ArchivedCells>(&self.cells_mmap) }
    }

    // --- Schema accessors ---

    #[inline]
    pub fn num_tables(&self) -> usize {
        self.archived_schema().tables.len()
    }

    #[inline]
    pub fn num_columns(&self) -> usize {
        self.archived_schema().columns.len()
    }

    #[inline]
    pub fn num_rows(&self) -> usize {
        self.archived_cells().row_offsets.len().saturating_sub(1)
    }

    #[inline]
    pub fn num_edges(&self) -> usize {
        self.archived_graph().outgoing.col_idx.len()
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    #[inline]
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    #[inline]
    pub fn table(&self, idx: TableIdx) -> &ArchivedTable {
        &self.archived_schema().tables[idx.0 as usize]
    }

    #[inline]
    pub fn column(&self, idx: ColumnIdx) -> &ArchivedColumn {
        &self.archived_schema().columns[idx.0 as usize]
    }

    #[inline]
    pub fn tables(&self) -> &[ArchivedTable] {
        &self.archived_schema().tables
    }

    #[inline]
    pub fn columns(&self) -> &[ArchivedColumn] {
        &self.archived_schema().columns
    }

    /// Get which table a row belongs to (binary search).
    #[inline]
    pub fn row_table(&self, row_idx: RowIdx) -> TableIdx {
        let row = row_idx.0;
        let tables = &self.archived_schema().tables;
        let idx = tables
            .binary_search_by(|table| {
                let range_start: u32 = table.row_range.0.0.into();
                let range_end: u32 = table.row_range.1.0.into();
                if row < range_start {
                    std::cmp::Ordering::Greater
                } else if row >= range_end {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .expect("Row not found in any table");
        let table_idx: u32 = tables[idx].idx.0.into();
        TableIdx(table_idx)
    }

    // --- Cell accessors ---

    #[inline]
    pub fn row_cells(&self, row_idx: RowIdx) -> &[PackedCell] {
        let cells = self.archived_cells();
        let start: u32 = cells.row_offsets[row_idx.0 as usize].into();
        let end: u32 = cells.row_offsets[row_idx.0 as usize + 1].into();
        unsafe {
            let archived_slice = &cells.cell_values[start as usize..end as usize];
            std::mem::transmute::<&[rkyv::rend::u32_le], &[u32]>(archived_slice)
        }
    }

    #[inline]
    pub fn row_timestamp(&self, row: RowIdx) -> Option<i64> {
        let ts: i64 = self.archived_cells().row_timestamps[row.0 as usize].into();
        if ts == NO_TIMESTAMP { None } else { Some(ts) }
    }

    // --- Graph accessors ---

    #[inline]
    pub fn outgoing_neighbors(&self, row: RowIdx) -> &[u32] {
        let graph = &self.archived_graph().outgoing;
        let start: u32 = graph.row_ptr[row.0 as usize].into();
        let end: u32 = graph.row_ptr[row.0 as usize + 1].into();
        unsafe {
            let archived_slice = &graph.col_idx[start as usize..end as usize];
            std::mem::transmute::<&[rkyv::rend::u32_le], &[u32]>(archived_slice)
        }
    }

    #[inline]
    pub fn incoming_neighbors(&self, row: RowIdx) -> &[u32] {
        let graph = &self.archived_graph().incoming;
        let start: u32 = graph.row_ptr[row.0 as usize].into();
        let end: u32 = graph.row_ptr[row.0 as usize + 1].into();
        unsafe {
            let archived_slice = &graph.col_idx[start as usize..end as usize];
            std::mem::transmute::<&[rkyv::rend::u32_le], &[u32]>(archived_slice)
        }
    }

    // --- Embedding accessors ---

    #[inline]
    pub fn get_embedding(&self, idx: EmbeddingIdx) -> &[f16] {
        let data_start = 8; // Skip header
        let start = data_start + idx.0 as usize * self.embed_dim * 2;
        let end = start + self.embed_dim * 2;
        unsafe {
            std::slice::from_raw_parts(
                self.embeddings_mmap[start..end].as_ptr() as *const f16,
                self.embed_dim,
            )
        }
    }

    #[inline]
    pub fn get_column_embedding(&self, idx: ColumnIdx) -> &[f16] {
        let schema = self.archived_schema();
        let embed_dim: u32 = schema.embed_dim.into();
        let embed_dim = embed_dim as usize;
        let start = idx.0 as usize * embed_dim;
        let end = start + embed_dim;
        let archived_slice = &schema.column_embeddings[start..end];
        unsafe {
            std::slice::from_raw_parts(archived_slice.as_ptr() as *const f16, archived_slice.len())
        }
    }

    // --- Timestamp stats ---

    #[inline]
    pub fn timestamp_mean(&self) -> Option<f64> {
        match &self.archived_schema().timestamp_mean {
            rkyv::option::ArchivedOption::Some(v) => Some((*v).into()),
            rkyv::option::ArchivedOption::None => None,
        }
    }

    #[inline]
    pub fn timestamp_std(&self) -> Option<f64> {
        match &self.archived_schema().timestamp_std {
            rkyv::option::ArchivedOption::Some(v) => Some((*v).into()),
            rkyv::option::ArchivedOption::None => None,
        }
    }
}

impl Clone for Database {
    fn clone(&self) -> Self {
        Self {
            schema_mmap: Arc::clone(&self.schema_mmap),
            graph_mmap: Arc::clone(&self.graph_mmap),
            cells_mmap: Arc::clone(&self.cells_mmap),
            embeddings_mmap: Arc::clone(&self.embeddings_mmap),
            embed_dim: self.embed_dim,
            vocab_size: self.vocab_size,
        }
    }
}

// ============================================================================
// Helper Traits for Archived Types
// ============================================================================

/// Extension methods for ArchivedTable.
pub trait ArchivedTableExt {
    fn cell_column(&self, cell_pos: usize) -> ColumnIdx;
    fn idx_native(&self) -> TableIdx;
    fn time_column_native(&self) -> Option<ColumnIdx>;
    fn feature_columns_slice(&self) -> &[ArchivedColumnIdx];
}

impl ArchivedTableExt for ArchivedTable {
    #[inline]
    fn cell_column(&self, cell_pos: usize) -> ColumnIdx {
        let archived_col_idx = &self.feature_columns[cell_pos];
        ColumnIdx(archived_col_idx.0.into())
    }

    #[inline]
    fn idx_native(&self) -> TableIdx {
        TableIdx(self.idx.0.into())
    }

    #[inline]
    fn time_column_native(&self) -> Option<ColumnIdx> {
        match &self.time_column {
            rkyv::option::ArchivedOption::Some(col_idx) => Some(ColumnIdx(col_idx.0.into())),
            rkyv::option::ArchivedOption::None => None,
        }
    }

    #[inline]
    fn feature_columns_slice(&self) -> &[ArchivedColumnIdx] {
        &self.feature_columns
    }
}

/// Extension methods for ArchivedColumn.
pub trait ArchivedColumnExt {
    fn stype_native(&self) -> SemanticType;
    fn is_prediction_target(&self) -> bool;
}

impl ArchivedColumnExt for ArchivedColumn {
    #[inline]
    fn stype_native(&self) -> SemanticType {
        match self.stype {
            ArchivedSemanticType::Numerical => SemanticType::Numerical,
            ArchivedSemanticType::Categorical => SemanticType::Categorical,
            ArchivedSemanticType::Text => SemanticType::Text,
            ArchivedSemanticType::Timestamp => SemanticType::Timestamp,
        }
    }

    #[inline]
    fn is_prediction_target(&self) -> bool {
        self.is_prediction_target
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
            feature_columns: vec![ColumnIdx(6), ColumnIdx(7), ColumnIdx(9)],
            row_range: (RowIdx(100), RowIdx(200)),
            primary_key_column: Some(ColumnIdx(5)),
            time_column: Some(ColumnIdx(9)),
        };

        assert!(table.contains_column(ColumnIdx(5)));
        assert!(table.contains_column(ColumnIdx(9)));
        assert!(!table.contains_column(ColumnIdx(10)));
        assert!(!table.contains_column(ColumnIdx(4)));

        assert!(table.contains_row(RowIdx(100)));
        assert!(table.contains_row(RowIdx(199)));
        assert!(!table.contains_row(RowIdx(200)));
        assert!(!table.contains_row(RowIdx(99)));

        assert_eq!(table.num_columns(), 5);
        assert_eq!(table.num_feature_columns(), 3);
        assert_eq!(table.cell_column(0), ColumnIdx(6));
        assert_eq!(table.cell_column(1), ColumnIdx(7));
        assert_eq!(table.cell_column(2), ColumnIdx(9));
    }

    #[test]
    fn test_csr_graph() {
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
        let idx3 = ctx.intern_text("hello");

        assert_eq!(idx1.0, 0);
        assert_eq!(idx2.0, 1);
        assert_eq!(idx3.0, 0);
        assert_eq!(ctx.vocab_size(), 2);

        ctx.register_pk(TableIdx(0), 42, RowIdx(100));
        assert_eq!(ctx.lookup_pk(TableIdx(0), 42), Some(RowIdx(100)));
        assert_eq!(ctx.lookup_pk(TableIdx(0), 99), None);
    }

    #[test]
    fn test_packed_cell_roundtrip() {
        let val = 3.14159f32;
        let packed = pack_numerical(val);
        assert_eq!(f32::from_bits(packed), val);

        let epoch = 1704067200.0f32;
        let packed = pack_timestamp(epoch);
        assert_eq!(f32::from_bits(packed), epoch);

        let idx = EmbeddingIdx(12345);
        let packed = pack_embedding_idx(idx);
        assert_eq!(packed, 12345);

        let packed = pack_null();
        assert!(is_packed_null(packed));
        assert!(!is_packed_null(0));
        assert!(!is_packed_null(pack_numerical(0.0)));
    }
}
