// Batch sampler for relational data.
// Performs BFS traversal of the FK graph starting from seed rows,
// producing flat vectors that get reshaped in Python for the model.
//
// Sequence filling is parallelized using Rayon for high throughput.

use crate::types::{
    ColumnIdx, Database, NormalizedCellValue, RawCellValue, RowIdx, SemanticType, TableIdx,
};
use half::f16;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::{IntoPyObjectExt, Py, PyAny};
use rand::prelude::*;
use rand::seq::index;
use rayon::prelude::*;
use std::path::Path;

/// A pointer wrapper that implements Send+Sync.
/// SAFETY: Only use when you guarantee non-overlapping access from different threads.
#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }

    unsafe fn add(self, offset: usize) -> *mut T {
        unsafe { self.0.add(offset) }
    }
}

/// Maximum number of foreign-to-primary neighbors we track per cell.
/// Must match MAX_F2P_NEIGHBORS in model.py
const MAX_F2P_NEIGHBORS: usize = 5;

// ============================================================================
// SequenceSlice: Mutable view into a single sequence's portion of BatchVecs
// ============================================================================

/// Mutable slices into one sequence's portion of BatchVecs.
/// Allows parallel filling directly into the final batch without intermediate allocation.
struct SequenceSlice<'a> {
    // Internal indices
    node_indices: &'a mut [i32],
    f2p_neighbor_indices: &'a mut [i32],
    table_name_indices: &'a mut [i32],
    column_name_indices: &'a mut [i32],
    // Values
    number_values: &'a mut [f16],
    datetime_values: &'a mut [f16],
    boolean_values: &'a mut [f16],
    text_values: &'a mut [f16],
    column_name_values: &'a mut [f16],
    // Types and flags
    semantic_types: &'a mut [i32],
    masks: &'a mut [bool],
    is_task_node: &'a mut [bool],
    is_padding: &'a mut [bool],
    // Attention masks for this sequence
    column_attn_mask: &'a mut [bool],
    feature_attn_mask: &'a mut [bool],
    neighbor_attn_mask: &'a mut [bool],
}

// ============================================================================
// BatchVecs: Flat vectors that will be reshaped in Python
// ============================================================================

/// Container for all batch data as flat vectors.
/// Python will reshape these to (batch_size, seq_len, ...) after receiving.
struct BatchVecs {
    // --- Internal indices (used for attention mask computation, not exported) ---
    /// Which row each position belongs to: for same-node attention
    node_indices: Vec<i32>,
    /// Foreign-to-primary neighbor row indices: (seq_len * MAX_F2P_NEIGHBORS)
    f2p_neighbor_indices: Vec<i32>,
    /// Table index for each position: for same-table checks
    table_name_indices: Vec<i32>,
    /// Column index for each position: for same-column attention
    column_name_indices: Vec<i32>,

    // --- Values (exported to Python) ---
    /// Z-score normalized numeric values
    number_values: Vec<f16>,
    /// Z-score normalized datetime values (global normalization)
    datetime_values: Vec<f16>,
    /// Z-score normalized boolean values
    boolean_values: Vec<f16>,
    /// Text embedding vectors: seq_len * d_text
    text_values: Vec<f16>,
    /// Column name embeddings: seq_len * d_text
    column_name_values: Vec<f16>,

    // --- Semantic types (exported) ---
    /// 0=Number, 1=Text, 2=Datetime, 3=Boolean
    semantic_types: Vec<i32>,

    // --- Flags (exported) ---
    /// Which positions are masked (for MLM-style training, also marks targets)
    masks: Vec<bool>,
    /// Does this position belong to a task table row?
    is_task_node: Vec<bool>,
    /// Is this a padding position?
    is_padding: Vec<bool>,

    // --- Precomputed attention masks (b * s * s, exported) ---
    /// Column attention: same column AND same table (excludes padding)
    column_attn_mask: Vec<bool>,
    /// Feature attention: same node OR kv is in q's f2p neighbors (excludes padding)
    feature_attn_mask: Vec<bool>,
    /// Neighbor attention: q is in kv's p2f neighbors (excludes padding)
    neighbor_attn_mask: Vec<bool>,

    /// Actual number of sequences (before padding to batch_size)
    true_batch_size: usize,
}

impl BatchVecs {
    /// Create a new BatchVecs with all vectors pre-allocated to correct size.
    /// All values are initialized to defaults (zeros, false, padding=true).
    fn new_preallocated(
        batch_size: usize,
        seq_len: usize,
        d_text: usize,
        true_batch_size: usize,
    ) -> Self {
        let l = batch_size * seq_len;
        let l_sq = batch_size * seq_len * seq_len;
        let l_f2p = l * MAX_F2P_NEIGHBORS;
        let l_text = l * d_text;

        Self {
            node_indices: vec![-1; l],
            f2p_neighbor_indices: vec![-1; l_f2p],
            table_name_indices: vec![0; l],
            column_name_indices: vec![0; l],
            number_values: vec![f16::ZERO; l],
            datetime_values: vec![f16::ZERO; l],
            boolean_values: vec![f16::ZERO; l],
            text_values: vec![f16::ZERO; l_text],
            column_name_values: vec![f16::ZERO; l_text],
            semantic_types: vec![0; l],
            masks: vec![false; l],
            is_task_node: vec![false; l],
            is_padding: vec![true; l], // Default to padding
            column_attn_mask: vec![false; l_sq],
            feature_attn_mask: vec![false; l_sq],
            neighbor_attn_mask: vec![false; l_sq],
            true_batch_size,
        }
    }

    /// Create with UNINITIALIZED memory - caller MUST fill all positions!
    /// This is much faster as it skips zeroing 193 MB of data.
    ///
    /// SAFETY: All positions must be written before reading.
    unsafe fn new_uninit(batch_size: usize, seq_len: usize, d_text: usize) -> Self {
        let l = batch_size * seq_len;
        let l_sq = batch_size * seq_len * seq_len;
        let l_f2p = l * MAX_F2P_NEIGHBORS;
        let l_text = l * d_text;

        // Helper to create uninitialized vec
        fn uninit_vec<T>(len: usize) -> Vec<T> {
            let mut v = Vec::with_capacity(len);
            unsafe { v.set_len(len) };
            v
        }

        Self {
            node_indices: uninit_vec(l),
            f2p_neighbor_indices: uninit_vec(l_f2p),
            table_name_indices: uninit_vec(l),
            column_name_indices: uninit_vec(l),
            number_values: uninit_vec(l),
            datetime_values: uninit_vec(l),
            boolean_values: uninit_vec(l),
            text_values: uninit_vec(l_text),
            column_name_values: uninit_vec(l_text),
            semantic_types: uninit_vec(l),
            masks: uninit_vec(l),
            is_task_node: uninit_vec(l),
            is_padding: uninit_vec(l),
            column_attn_mask: uninit_vec(l_sq),
            feature_attn_mask: uninit_vec(l_sq),
            neighbor_attn_mask: uninit_vec(l_sq),
            true_batch_size: batch_size,
        }
    }

    /// Convert to Python objects (list of (name, numpy_array) tuples).
    /// Only exports fields that match the Python Batch dataclass.
    fn into_pyobject(self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        // Note: node_indices, f2p_neighbor_indices, table_name_indices, column_name_indices
        // are intentionally NOT exported - they were only needed internally for mask computation.
        Ok(vec![
            // Values
            ("number_values", PyArray1::from_vec(py, self.number_values)).into_py_any(py)?,
            (
                "datetime_values",
                PyArray1::from_vec(py, self.datetime_values),
            )
                .into_py_any(py)?,
            (
                "boolean_values",
                PyArray1::from_vec(py, self.boolean_values),
            )
                .into_py_any(py)?,
            ("text_values", PyArray1::from_vec(py, self.text_values)).into_py_any(py)?,
            (
                "column_name_values",
                PyArray1::from_vec(py, self.column_name_values),
            )
                .into_py_any(py)?,
            // Semantic types
            (
                "semantic_types",
                PyArray1::from_vec(py, self.semantic_types),
            )
                .into_py_any(py)?,
            // Flags
            ("masks", PyArray1::from_vec(py, self.masks)).into_py_any(py)?,
            ("is_task_node", PyArray1::from_vec(py, self.is_task_node)).into_py_any(py)?,
            ("is_padding", PyArray1::from_vec(py, self.is_padding)).into_py_any(py)?,
            // Precomputed attention masks (b * s * s)
            (
                "column_attn_mask",
                PyArray1::from_vec(py, self.column_attn_mask),
            )
                .into_py_any(py)?,
            (
                "feature_attn_mask",
                PyArray1::from_vec(py, self.feature_attn_mask),
            )
                .into_py_any(py)?,
            (
                "neighbor_attn_mask",
                PyArray1::from_vec(py, self.neighbor_attn_mask),
            )
                .into_py_any(py)?,
            // Metadata
            ("true_batch_size", self.true_batch_size).into_py_any(py)?,
        ])
    }
}

// ============================================================================
// BatchBuffer: Reusable buffer for zero-allocation batch generation
// ============================================================================

/// Reusable buffer for batch generation.
/// Create once, reuse across batches to avoid allocation overhead.
/// This eliminates the 73% allocation overhead seen in profiling.
pub struct BatchBuffer {
    vecs: BatchVecs,
    batch_size: usize,
    seq_len: usize,
    d_text: usize,
}

impl BatchBuffer {
    /// Create a new reusable buffer. This allocates once.
    pub fn new(batch_size: usize, seq_len: usize, d_text: usize) -> Self {
        // Use uninit since we'll fill everything on each use
        let vecs = unsafe { BatchVecs::new_uninit(batch_size, seq_len, d_text) };
        Self {
            vecs,
            batch_size,
            seq_len,
            d_text,
        }
    }

    /// Reset buffer for reuse. Only zeros the minimal required fields.
    /// - is_padding: set all to true (default state)
    /// - attention masks: no reset needed (fully overwritten)
    /// - values: no reset needed (only non-padding positions are read)
    fn reset(&mut self) {
        // Only is_padding needs reset - set all to true
        // This is ~200x faster than zeroing everything (32KB vs 193MB)
        self.vecs.is_padding.fill(true);
    }

    /// Get the internal BatchVecs (transfers ownership for Python export)
    #[allow(dead_code)]
    fn take_vecs(&mut self) -> BatchVecs {
        // Create a new uninit buffer and swap
        let new_vecs = unsafe { BatchVecs::new_uninit(self.batch_size, self.seq_len, self.d_text) };
        std::mem::replace(&mut self.vecs, new_vecs)
    }
}

// ============================================================================
// SamplerItem: A seed row for BFS sampling
// ============================================================================

/// Represents a single training example (seed row for BFS)
struct SamplerItem {
    /// Which database this item comes from
    database_idx: usize,
    /// Which row to start BFS from
    row_idx: RowIdx,
    /// Which column is the target for prediction
    target_column: ColumnIdx,
    /// Columns to drop from the seed row (prevent target leakage)
    columns_to_drop: Vec<ColumnIdx>,
    /// Whether this is a task table row (train/val/test split)
    is_task_table: bool,
}

// ============================================================================
// Sampler: Main PyO3 class
// ============================================================================

#[pyclass]
pub struct Sampler {
    /// All loaded databases (indexed by SamplerItem.database_idx)
    databases: Vec<Database>,
    /// All items available for sampling (from all databases)
    items: Vec<SamplerItem>,
    /// Batch size (number of sequences per batch)
    batch_size: usize,
    /// Sequence length (number of cells per sequence)
    seq_len: usize,
    /// Distributed training rank
    rank: usize,
    /// Distributed training world size
    world_size: usize,
    /// Max neighbors to sample at each BFS step (prevents explosion)
    max_bfs_width: usize,
    /// Text embedding dimension
    d_text: usize,
    /// Random seed
    seed: u64,
    /// Current epoch (for deterministic shuffling)
    epoch: u64,
}

#[pymethods]
impl Sampler {
    /// Create a new Sampler for multi-database training.
    ///
    /// Args:
    ///     db_configs: List of (db_path, task_table_idx, target_column_idx, columns_to_drop)
    ///                 tuples. Each database will be loaded and its task table rows added
    ///                 to the sampling pool. Items from all databases are shuffled together.
    ///     batch_size: Number of sequences per batch
    ///     seq_len: Maximum sequence length (cells per sequence)
    ///     rank: Distributed training rank (0 for single GPU)
    ///     world_size: Number of distributed workers (1 for single GPU)
    ///     max_bfs_width: Max neighbors sampled per BFS step
    ///     d_text: Text embedding dimension
    ///     seed: Random seed for reproducibility
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        db_configs: Vec<(String, u32, u32, Vec<u32>)>,
        batch_size: usize,
        seq_len: usize,
        rank: usize,
        world_size: usize,
        max_bfs_width: usize,
        d_text: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let mut databases = Vec::new();
        let mut items = Vec::new();

        for (db_idx, (db_path, task_table_idx, target_column_idx, cols_to_drop)) in
            db_configs.into_iter().enumerate()
        {
            // Load the database
            let database = Database::load(Path::new(&db_path)).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "Failed to load database '{}': {}",
                    db_path, e
                ))
            })?;

            let task_table = TableIdx(task_table_idx);
            let target_column = ColumnIdx(target_column_idx);
            let columns_to_drop: Vec<ColumnIdx> = cols_to_drop.into_iter().map(ColumnIdx).collect();

            // Create items from all rows in the task table
            let table = database.get_table(task_table);
            for row_idx in table.row_range.0.0..table.row_range.1.0 {
                items.push(SamplerItem {
                    database_idx: db_idx,
                    row_idx: RowIdx(row_idx),
                    target_column,
                    columns_to_drop: columns_to_drop.clone(),
                    is_task_table: true,
                });
            }

            databases.push(database);
        }

        Ok(Self {
            databases,
            items,
            batch_size,
            seq_len,
            rank,
            world_size,
            max_bfs_width,
            d_text,
            seed,
            epoch: 0,
        })
    }

    /// Number of batches per epoch
    fn len_py(&self) -> usize {
        self.len()
    }

    /// Get a batch by index, returns list of (name, numpy_array) tuples
    fn batch_py(&self, py: Python<'_>, batch_idx: usize) -> PyResult<Vec<Py<PyAny>>> {
        self.batch(batch_idx).into_pyobject(py)
    }

    /// Shuffle items for a new epoch
    fn shuffle_py(&mut self, epoch: u64) {
        self.epoch = epoch;
        let mut rng = StdRng::seed_from_u64(epoch.wrapping_add(self.seed));
        self.items.shuffle(&mut rng);
    }
}

// Rust-only methods
impl Sampler {
    /// Number of batches per epoch (accounting for distributed training)
    fn len(&self) -> usize {
        self.items.len().div_ceil(self.batch_size * self.world_size)
    }

    /// Create a reusable batch buffer for this sampler's configuration.
    fn create_buffer(&self) -> BatchBuffer {
        BatchBuffer::new(self.batch_size, self.seq_len, self.d_text)
    }

    /// Generate a batch of sequences (allocates new BatchVecs).
    /// For training loops, prefer `batch_into` with a reusable buffer.
    fn batch(&self, batch_idx: usize) -> BatchVecs {
        let start_idx = batch_idx * self.batch_size * self.world_size + self.rank * self.batch_size;
        let true_batch_size = self
            .batch_size
            .min(self.items.len().saturating_sub(start_idx));

        // Use uninit for speed - we fill everything
        let mut vecs = unsafe { BatchVecs::new_uninit(self.batch_size, self.seq_len, self.d_text) };
        vecs.true_batch_size = true_batch_size;

        // Initialize is_padding to true (only field that needs default)
        vecs.is_padding.fill(true);

        self.fill_batch_vecs(&mut vecs, start_idx);
        vecs
    }

    /// Fill a batch into a reusable buffer (zero allocation after warmup).
    /// This is ~3x faster than `batch()` due to buffer reuse.
    fn batch_into(&self, buffer: &mut BatchBuffer, batch_idx: usize) {
        let start_idx = batch_idx * self.batch_size * self.world_size + self.rank * self.batch_size;
        let true_batch_size = self
            .batch_size
            .min(self.items.len().saturating_sub(start_idx));

        buffer.reset();
        buffer.vecs.true_batch_size = true_batch_size;
        self.fill_batch_vecs(&mut buffer.vecs, start_idx);
    }

    /// Internal: fill a BatchVecs with data (shared between batch and batch_into)
    fn fill_batch_vecs(&self, vecs: &mut BatchVecs, start_idx: usize) {
        // Get raw pointers for parallel mutable access to non-overlapping regions
        // SAFETY: Each sequence writes to a disjoint slice, no overlap
        let node_ptr = SyncPtr::new(vecs.node_indices.as_mut_ptr());
        let f2p_ptr = SyncPtr::new(vecs.f2p_neighbor_indices.as_mut_ptr());
        let table_ptr = SyncPtr::new(vecs.table_name_indices.as_mut_ptr());
        let col_ptr = SyncPtr::new(vecs.column_name_indices.as_mut_ptr());
        let num_ptr = SyncPtr::new(vecs.number_values.as_mut_ptr());
        let dt_ptr = SyncPtr::new(vecs.datetime_values.as_mut_ptr());
        let bool_ptr = SyncPtr::new(vecs.boolean_values.as_mut_ptr());
        let text_ptr = SyncPtr::new(vecs.text_values.as_mut_ptr());
        let colname_ptr = SyncPtr::new(vecs.column_name_values.as_mut_ptr());
        let sem_ptr = SyncPtr::new(vecs.semantic_types.as_mut_ptr());
        let mask_ptr = SyncPtr::new(vecs.masks.as_mut_ptr());
        let task_ptr = SyncPtr::new(vecs.is_task_node.as_mut_ptr());
        let pad_ptr = SyncPtr::new(vecs.is_padding.as_mut_ptr());
        let col_attn_ptr = SyncPtr::new(vecs.column_attn_mask.as_mut_ptr());
        let feat_attn_ptr = SyncPtr::new(vecs.feature_attn_mask.as_mut_ptr());
        let nbr_attn_ptr = SyncPtr::new(vecs.neighbor_attn_mask.as_mut_ptr());

        let seq_len = self.seq_len;
        let d_text = self.d_text;
        let items_len = self.items.len();

        // Fill sequences AND compute attention masks in parallel
        (0..self.batch_size).into_par_iter().for_each(|i| {
            let j = (start_idx + i) % items_len;
            let item = &self.items[j];

            // Calculate offsets for this sequence
            let seq_offset = i * seq_len;
            let f2p_offset = i * seq_len * MAX_F2P_NEIGHBORS;
            let text_offset = i * seq_len * d_text;
            let mask_offset = i * seq_len * seq_len;

            // SAFETY: Each iteration writes to disjoint regions
            unsafe {
                let slice = SequenceSlice {
                    node_indices: std::slice::from_raw_parts_mut(node_ptr.add(seq_offset), seq_len),
                    f2p_neighbor_indices: std::slice::from_raw_parts_mut(
                        f2p_ptr.add(f2p_offset),
                        seq_len * MAX_F2P_NEIGHBORS,
                    ),
                    table_name_indices: std::slice::from_raw_parts_mut(
                        table_ptr.add(seq_offset),
                        seq_len,
                    ),
                    column_name_indices: std::slice::from_raw_parts_mut(
                        col_ptr.add(seq_offset),
                        seq_len,
                    ),
                    number_values: std::slice::from_raw_parts_mut(num_ptr.add(seq_offset), seq_len),
                    datetime_values: std::slice::from_raw_parts_mut(
                        dt_ptr.add(seq_offset),
                        seq_len,
                    ),
                    boolean_values: std::slice::from_raw_parts_mut(
                        bool_ptr.add(seq_offset),
                        seq_len,
                    ),
                    text_values: std::slice::from_raw_parts_mut(
                        text_ptr.add(text_offset),
                        seq_len * d_text,
                    ),
                    column_name_values: std::slice::from_raw_parts_mut(
                        colname_ptr.add(text_offset),
                        seq_len * d_text,
                    ),
                    semantic_types: std::slice::from_raw_parts_mut(
                        sem_ptr.add(seq_offset),
                        seq_len,
                    ),
                    masks: std::slice::from_raw_parts_mut(mask_ptr.add(seq_offset), seq_len),
                    is_task_node: std::slice::from_raw_parts_mut(task_ptr.add(seq_offset), seq_len),
                    is_padding: std::slice::from_raw_parts_mut(pad_ptr.add(seq_offset), seq_len),
                    column_attn_mask: std::slice::from_raw_parts_mut(
                        col_attn_ptr.add(mask_offset),
                        seq_len * seq_len,
                    ),
                    feature_attn_mask: std::slice::from_raw_parts_mut(
                        feat_attn_ptr.add(mask_offset),
                        seq_len * seq_len,
                    ),
                    neighbor_attn_mask: std::slice::from_raw_parts_mut(
                        nbr_attn_ptr.add(mask_offset),
                        seq_len * seq_len,
                    ),
                };

                self.fill_sequence_inline(item, slice);
            }
        });
    }

    /// Fill one sequence via BFS AND compute its attention masks inline.
    /// Writes directly to the provided slices (no intermediate allocation).
    fn fill_sequence_inline(&self, item: &SamplerItem, mut seq: SequenceSlice<'_>) {
        let db = &self.databases[item.database_idx];
        let seed_row = db.get_row(item.row_idx);
        let seed_table = db.get_table(seed_row.table_idx);

        // Get timestamp from seed row for temporal filtering
        let seed_timestamp = self.get_row_timestamp(db, item.row_idx);

        // BFS state
        let mut visited = vec![false; db.num_rows()];

        // Two frontiers: foreign-to-primary (follow FK to parent) and primary-to-foreign (referenced by)
        // f2p gets priority (more semantically relevant)
        let mut f2p_frontier: Vec<(usize, RowIdx)> = vec![(0, item.row_idx)];
        let mut p2f_frontier: Vec<Vec<RowIdx>> = Vec::new();

        let mut seq_i = 0;
        let mut rng = StdRng::seed_from_u64(
            self.epoch
                .wrapping_add(item.row_idx.0 as u64)
                .wrapping_add(self.seed),
        );

        loop {
            // Select next node: prioritize f2p (parent) edges
            let (depth, row_idx) = if let Some(frontier_item) = f2p_frontier.pop() {
                frontier_item
            } else {
                // Find first non-empty depth in p2f frontier
                let mut found = None;
                for (d, nodes) in p2f_frontier.iter().enumerate() {
                    if !nodes.is_empty() {
                        found = Some(d);
                        break;
                    }
                }
                if let Some(d) = found {
                    let r = rng.random_range(0..p2f_frontier[d].len());
                    let len = p2f_frontier[d].len();
                    p2f_frontier[d].swap(r, len - 1);
                    let row_idx = p2f_frontier[d].pop().unwrap();
                    (d, row_idx)
                } else {
                    // No more nodes to visit
                    break;
                }
            };

            // Skip if already visited
            if visited[row_idx.0 as usize] {
                continue;
            }
            visited[row_idx.0 as usize] = true;

            let row = db.get_row(row_idx);
            let table = db.get_table(row.table_idx);

            // Collect f2p neighbors for this row (rows this row points TO via FK)
            let mut f2p_neighbors: Vec<RowIdx> = Vec::new();

            // Follow foreign key edges (f2p: this row -> parent rows)
            for &edge_idx in &db.edges_from[row_idx.0 as usize] {
                let edge = &db.fk_edges[edge_idx];
                f2p_neighbors.push(edge.to_row);
                f2p_frontier.push((depth + 1, edge.to_row));
            }

            // Follow reverse edges (p2f: child rows -> this row)
            let mut db_children: Vec<RowIdx> = Vec::new();
            for &edge_idx in &db.edges_to[row_idx.0 as usize] {
                let edge = &db.fk_edges[edge_idx];
                let child_row = db.get_row(edge.from_row);

                // Temporal constraint: don't include future rows
                if let Some(cutoff) = seed_timestamp
                    && !db.row_is_before(edge.from_row, cutoff)
                {
                    continue;
                }

                // Only include task table edges if seed is from task table
                let child_table = db.get_table(child_row.table_idx);
                if child_table.idx == seed_table.idx || item.is_task_table {
                    db_children.push(edge.from_row);
                }
            }

            // Subsample if too many children (prevents sequence explosion)
            let sampled_children = if db_children.len() > self.max_bfs_width {
                let idxs =
                    index::sample(&mut rng, db_children.len(), self.max_bfs_width).into_vec();
                idxs.into_iter().map(|i| db_children[i]).collect()
            } else {
                db_children
            };

            // Add sampled children to p2f frontier
            for child_row in sampled_children {
                while p2f_frontier.len() <= depth + 1 {
                    p2f_frontier.push(Vec::new());
                }
                p2f_frontier[depth + 1].push(child_row);
            }

            // Now fill in cells from this row
            let col_start = table.column_range.0.0;
            let is_seed_row = row_idx == item.row_idx;

            for (local_idx, normalized_cell) in row.normalized.iter().enumerate() {
                let global_col_idx = ColumnIdx(col_start + local_idx as u32);
                let column = db.get_column(global_col_idx);

                // Skip columns that should be dropped (prevent leakage)
                if is_seed_row && item.columns_to_drop.contains(&global_col_idx) {
                    continue;
                }

                // Skip null values
                if matches!(normalized_cell, NormalizedCellValue::Null) {
                    continue;
                }

                // Fill the position
                seq.node_indices[seq_i] = row_idx.0 as i32;
                seq.table_name_indices[seq_i] = row.table_idx.0 as i32;
                seq.column_name_indices[seq_i] = global_col_idx.0 as i32;

                // Fill f2p neighbor indices (up to MAX_F2P_NEIGHBORS)
                for (j, &neighbor_row) in f2p_neighbors.iter().take(MAX_F2P_NEIGHBORS).enumerate() {
                    seq.f2p_neighbor_indices[seq_i * MAX_F2P_NEIGHBORS + j] = neighbor_row.0 as i32;
                }

                // Semantic type
                seq.semantic_types[seq_i] = column.dtype as i32;

                // Fill values based on semantic type
                match (normalized_cell, column.dtype) {
                    (NormalizedCellValue::Scalar(v), SemanticType::Number) => {
                        seq.number_values[seq_i] = f16::from_f32(*v);
                    }
                    (NormalizedCellValue::Scalar(v), SemanticType::Datetime) => {
                        seq.datetime_values[seq_i] = f16::from_f32(*v);
                    }
                    (NormalizedCellValue::Scalar(v), SemanticType::Boolean) => {
                        seq.boolean_values[seq_i] = f16::from_f32(*v);
                    }
                    (NormalizedCellValue::Text(text_idx), SemanticType::Text) => {
                        // Copy text embedding
                        if let Some(embedding) = db.text_value_embeddings.get(text_idx.0 as usize) {
                            let start = seq_i * self.d_text;
                            let end = start + self.d_text.min(embedding.len());
                            seq.text_values[start..end].copy_from_slice(&embedding[..end - start]);
                        }
                    }
                    _ => {}
                }

                // Copy column name embedding
                if let Some(ref embedding) = column.column_description_embedding {
                    let start = seq_i * self.d_text;
                    let end = start + self.d_text.min(embedding.len());
                    seq.column_name_values[start..end].copy_from_slice(&embedding[..end - start]);
                }

                // Mask the target position for prediction
                let is_target = is_seed_row && global_col_idx == item.target_column;
                seq.masks[seq_i] = is_target;

                // Is this a task table row?
                seq.is_task_node[seq_i] =
                    item.is_task_table && (row.table_idx == db.get_row(item.row_idx).table_idx);

                seq.is_padding[seq_i] = false; // Not padding

                seq_i += 1;
                if seq_i >= self.seq_len {
                    break;
                }
            }

            if seq_i >= self.seq_len {
                break;
            }
        }
        // Remaining positions stay as padding (is_padding = true by default)

        // Compute attention masks inline for this sequence
        self.compute_sequence_attention_masks(&mut seq);
    }

    /// Compute attention masks for a single sequence (inline, no separate pass).
    /// Uses AVX2 SIMD when available.
    #[cfg(target_arch = "x86_64")]
    fn compute_sequence_attention_masks(&self, seq: &mut SequenceSlice<'_>) {
        use std::arch::x86_64::*;

        let seq_len = self.seq_len;

        unsafe {
            let minus_one = _mm256_set1_epi32(-1);

            for q in 0..seq_len {
                // Skip if query is padding
                if seq.is_padding[q] {
                    continue;
                }

                let q_node = seq.node_indices[q];
                let q_table = seq.table_name_indices[q];
                let q_col = seq.column_name_indices[q];

                // Broadcast q values
                let q_node_v = _mm256_set1_epi32(q_node);
                let q_table_v = _mm256_set1_epi32(q_table);
                let q_col_v = _mm256_set1_epi32(q_col);

                // Get q's f2p neighbors
                let q_f2p_start = q * MAX_F2P_NEIGHBORS;
                let q_f2p: [__m256i; MAX_F2P_NEIGHBORS] = std::array::from_fn(|i| {
                    _mm256_set1_epi32(seq.f2p_neighbor_indices[q_f2p_start + i])
                });

                let mask_row_start = q * seq_len;
                let mut kv = 0;
                let simd_len = seq_len / 8 * 8;

                // SIMD loop (8 at a time)
                while kv < simd_len {
                    let kv_nodes =
                        _mm256_loadu_si256(seq.node_indices.as_ptr().add(kv) as *const __m256i);
                    let kv_tables = _mm256_loadu_si256(
                        seq.table_name_indices.as_ptr().add(kv) as *const __m256i
                    );
                    let kv_cols = _mm256_loadu_si256(
                        seq.column_name_indices.as_ptr().add(kv) as *const __m256i
                    );

                    // Column attention: same col AND same table
                    let col_eq = _mm256_cmpeq_epi32(q_col_v, kv_cols);
                    let table_eq = _mm256_cmpeq_epi32(q_table_v, kv_tables);
                    let col_result = _mm256_and_si256(col_eq, table_eq);

                    // Feature attention: same node OR kv in q's f2p
                    let same_node = _mm256_cmpeq_epi32(q_node_v, kv_nodes);
                    let mut in_f2p = _mm256_setzero_si256();
                    for i in 0..MAX_F2P_NEIGHBORS {
                        let valid = _mm256_cmpgt_epi32(q_f2p[i], minus_one);
                        let matches = _mm256_cmpeq_epi32(q_f2p[i], kv_nodes);
                        in_f2p = _mm256_or_si256(in_f2p, _mm256_and_si256(valid, matches));
                    }
                    let feat_result = _mm256_or_si256(same_node, in_f2p);

                    // Neighbor attention: q_node in kv's f2p
                    let indices = _mm256_set_epi32(
                        ((kv + 7) * MAX_F2P_NEIGHBORS) as i32,
                        ((kv + 6) * MAX_F2P_NEIGHBORS) as i32,
                        ((kv + 5) * MAX_F2P_NEIGHBORS) as i32,
                        ((kv + 4) * MAX_F2P_NEIGHBORS) as i32,
                        ((kv + 3) * MAX_F2P_NEIGHBORS) as i32,
                        ((kv + 2) * MAX_F2P_NEIGHBORS) as i32,
                        ((kv + 1) * MAX_F2P_NEIGHBORS) as i32,
                        (kv * MAX_F2P_NEIGHBORS) as i32,
                    );
                    let mut nbr_result = _mm256_setzero_si256();
                    for i in 0..MAX_F2P_NEIGHBORS {
                        let offset_indices = _mm256_add_epi32(indices, _mm256_set1_epi32(i as i32));
                        let kv_neighbors = _mm256_i32gather_epi32::<4>(
                            seq.f2p_neighbor_indices.as_ptr(),
                            offset_indices,
                        );
                        let matches = _mm256_cmpeq_epi32(kv_neighbors, q_node_v);
                        nbr_result = _mm256_or_si256(nbr_result, matches);
                    }

                    // Extract and store results
                    let col_bits = _mm256_movemask_epi8(col_result) as u32;
                    let feat_bits = _mm256_movemask_epi8(feat_result) as u32;
                    let nbr_bits = _mm256_movemask_epi8(nbr_result) as u32;

                    for lane in 0..8 {
                        if !seq.is_padding[kv + lane] {
                            let mask_idx = mask_row_start + kv + lane;
                            let bit_pos = lane * 4;
                            seq.column_attn_mask[mask_idx] = (col_bits >> bit_pos) & 0xF != 0;
                            seq.feature_attn_mask[mask_idx] = (feat_bits >> bit_pos) & 0xF != 0;
                            seq.neighbor_attn_mask[mask_idx] = (nbr_bits >> bit_pos) & 0xF != 0;
                        }
                    }

                    kv += 8;
                }

                // Scalar fallback for remaining
                while kv < seq_len {
                    if !seq.is_padding[kv] {
                        let kv_node = seq.node_indices[kv];
                        let kv_table = seq.table_name_indices[kv];
                        let kv_col = seq.column_name_indices[kv];
                        let mask_idx = mask_row_start + kv;

                        seq.column_attn_mask[mask_idx] = (q_col == kv_col) && (q_table == kv_table);

                        let same_node = q_node == kv_node;
                        let kv_in_q_f2p = (0..MAX_F2P_NEIGHBORS).any(|i| {
                            let n = seq.f2p_neighbor_indices[q_f2p_start + i];
                            n >= 0 && n == kv_node
                        });
                        seq.feature_attn_mask[mask_idx] = same_node || kv_in_q_f2p;

                        let kv_f2p_start = kv * MAX_F2P_NEIGHBORS;
                        let q_in_kv_f2p = (0..MAX_F2P_NEIGHBORS)
                            .any(|i| seq.f2p_neighbor_indices[kv_f2p_start + i] == q_node);
                        seq.neighbor_attn_mask[mask_idx] = q_in_kv_f2p;
                    }
                    kv += 1;
                }
            }
        }
    }

    /// Fallback for non-x86_64 architectures
    #[cfg(not(target_arch = "x86_64"))]
    fn compute_sequence_attention_masks(&self, seq: &mut SequenceSlice<'_>) {
        let seq_len = self.seq_len;

        for q in 0..seq_len {
            if seq.is_padding[q] {
                continue;
            }

            let q_node = seq.node_indices[q];
            let q_table = seq.table_name_indices[q];
            let q_col = seq.column_name_indices[q];
            let q_f2p_start = q * MAX_F2P_NEIGHBORS;

            for kv in 0..seq_len {
                if seq.is_padding[kv] {
                    continue;
                }

                let kv_node = seq.node_indices[kv];
                let kv_table = seq.table_name_indices[kv];
                let kv_col = seq.column_name_indices[kv];
                let mask_idx = q * seq_len + kv;

                seq.column_attn_mask[mask_idx] = (q_col == kv_col) && (q_table == kv_table);

                let same_node = q_node == kv_node;
                let kv_in_q_f2p = (0..MAX_F2P_NEIGHBORS).any(|i| {
                    let n = seq.f2p_neighbor_indices[q_f2p_start + i];
                    n >= 0 && n == kv_node
                });
                seq.feature_attn_mask[mask_idx] = same_node || kv_in_q_f2p;

                let kv_f2p_start = kv * MAX_F2P_NEIGHBORS;
                let q_in_kv_f2p = (0..MAX_F2P_NEIGHBORS)
                    .any(|i| seq.f2p_neighbor_indices[kv_f2p_start + i] == q_node);
                seq.neighbor_attn_mask[mask_idx] = q_in_kv_f2p;
            }
        }
    }

    /// Get the timestamp from a row's time column (if it has one)
    fn get_row_timestamp(&self, db: &Database, row_idx: RowIdx) -> Option<f32> {
        let row = db.get_row(row_idx);
        let table = db.get_table(row.table_idx);

        let time_col = table.time_col?;
        let local_idx = (time_col.0 - table.column_range.0.0) as usize;

        match row.raw.get(local_idx) {
            Some(RawCellValue::Datetime(ts)) => Some(*ts),
            _ => None,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Create a Sampler directly for testing (bypasses PyO3)
    fn create_test_sampler(
        db_configs: Vec<(String, u32, u32, Vec<u32>)>,
        batch_size: usize,
        seq_len: usize,
        d_text: usize,
    ) -> Sampler {
        let mut databases = Vec::new();
        let mut items = Vec::new();

        for (db_idx, (db_path, task_table_idx, target_column_idx, cols_to_drop)) in
            db_configs.into_iter().enumerate()
        {
            let database = Database::load(Path::new(&db_path)).expect("Failed to load database");

            let task_table = TableIdx(task_table_idx);
            let target_column = ColumnIdx(target_column_idx);
            let columns_to_drop: Vec<ColumnIdx> = cols_to_drop.into_iter().map(ColumnIdx).collect();

            let table = database.get_table(task_table);
            for row_idx in table.row_range.0.0..table.row_range.1.0 {
                items.push(SamplerItem {
                    database_idx: db_idx,
                    row_idx: RowIdx(row_idx),
                    target_column,
                    columns_to_drop: columns_to_drop.clone(),
                    is_task_table: true,
                });
            }

            databases.push(database);
        }

        Sampler {
            databases,
            items,
            batch_size,
            seq_len,
            rank: 0,
            world_size: 1,
            max_bfs_width: 256,
            d_text,
            seed: 42,
            epoch: 0,
        }
    }

    #[test]
    fn test_sampler_batch_generation() {
        let db_path = "sample_data_f1.rkyv";
        if !Path::new(db_path).exists() {
            eprintln!(
                "Skipping test: {} not found. Run preprocessor first.",
                db_path
            );
            return;
        }

        // Use the results table (26080 rows) with position as target
        // Table 7 = results, column 51 = position
        let batch_size = 32;
        let seq_len = 1024;
        let d_text = 768; // From the database embeddings

        println!("\n=== Creating Sampler ===");
        let start = Instant::now();
        let sampler = create_test_sampler(
            vec![(db_path.to_string(), 7, 51, vec![])],
            batch_size,
            seq_len,
            d_text,
        );
        println!("Sampler created in {:?}", start.elapsed());
        println!("Total items: {}", sampler.items.len());
        println!("Batches per epoch: {}", sampler.len());

        // Generate a batch
        println!("\n=== Generating Batch ===");
        let start = Instant::now();
        let batch = sampler.batch(0);
        let batch_time = start.elapsed();
        println!("Batch generated in {:?}", batch_time);

        // Verify batch structure
        let expected_len = batch_size * seq_len;
        assert_eq!(batch.number_values.len(), expected_len);
        assert_eq!(batch.datetime_values.len(), expected_len);
        assert_eq!(batch.boolean_values.len(), expected_len);
        assert_eq!(batch.text_values.len(), expected_len * d_text);
        assert_eq!(batch.column_name_values.len(), expected_len * d_text);
        assert_eq!(batch.semantic_types.len(), expected_len);
        assert_eq!(batch.masks.len(), expected_len);
        assert_eq!(batch.is_task_node.len(), expected_len);
        assert_eq!(batch.is_padding.len(), expected_len);

        // Check attention masks
        let expected_mask_len = batch_size * seq_len * seq_len;
        assert_eq!(batch.column_attn_mask.len(), expected_mask_len);
        assert_eq!(batch.feature_attn_mask.len(), expected_mask_len);
        assert_eq!(batch.neighbor_attn_mask.len(), expected_mask_len);

        assert_eq!(batch.true_batch_size, batch_size);

        // Count non-padding positions
        let non_padding: usize = batch.is_padding.iter().filter(|&&p| !p).count();
        let padding: usize = batch.is_padding.iter().filter(|&&p| p).count();
        println!(
            "Non-padding positions: {} ({:.1}%)",
            non_padding,
            100.0 * non_padding as f64 / expected_len as f64
        );
        println!("Padding positions: {}", padding);

        // Count masked positions (targets)
        let masked: usize = batch.masks.iter().filter(|&&m| m).count();
        println!("Masked (target) positions: {}", masked);
        assert!(
            masked > 0,
            "Should have at least one masked position per sequence"
        );

        // Verify attention masks have some true values
        let col_attn_count: usize = batch.column_attn_mask.iter().filter(|&&m| m).count();
        let feat_attn_count: usize = batch.feature_attn_mask.iter().filter(|&&m| m).count();
        let nbr_attn_count: usize = batch.neighbor_attn_mask.iter().filter(|&&m| m).count();
        println!("Column attention edges: {}", col_attn_count);
        println!("Feature attention edges: {}", feat_attn_count);
        println!("Neighbor attention edges: {}", nbr_attn_count);

        println!("\n=== Batch Generation Performance ===");
        println!(
            "Time per sequence: {:.2} ms",
            batch_time.as_secs_f64() * 1000.0 / batch_size as f64
        );
        println!(
            "Sequences per second: {:.0}",
            batch_size as f64 / batch_time.as_secs_f64()
        );

        println!("\n=== Test Passed ===");
    }

    #[test]
    fn test_sampler_multiple_batches() {
        let db_path = "sample_data_f1.rkyv";
        if !Path::new(db_path).exists() {
            return;
        }

        let batch_size = 32;
        let seq_len = 1024;
        let d_text = 768;

        let mut sampler = create_test_sampler(
            vec![(db_path.to_string(), 7, 51, vec![])],
            batch_size,
            seq_len,
            d_text,
        );

        // Shuffle for epoch 1
        sampler.epoch = 1;
        let mut rng = StdRng::seed_from_u64(1u64.wrapping_add(sampler.seed));
        sampler.items.shuffle(&mut rng);

        // Generate multiple batches
        let num_batches = 10.min(sampler.len());
        println!("\n=== Generating {} batches ===", num_batches);

        let start = Instant::now();
        for batch_idx in 0..num_batches {
            let batch = sampler.batch(batch_idx);
            assert_eq!(batch.number_values.len(), batch_size * seq_len);
        }
        let total_time = start.elapsed();

        println!("Total time: {:?}", total_time);
        println!(
            "Time per batch: {:.2} ms",
            total_time.as_secs_f64() * 1000.0 / num_batches as f64
        );
        println!(
            "Batches per second: {:.1}",
            num_batches as f64 / total_time.as_secs_f64()
        );
    }

    #[test]
    fn test_profile_batch_generation() {
        let db_path = "sample_data_f1.rkyv";
        if !Path::new(db_path).exists() {
            eprintln!("Skipping: {} not found", db_path);
            return;
        }

        let batch_size = 32;
        let seq_len = 1024;
        let d_text = 768;

        let sampler = create_test_sampler(
            vec![(db_path.to_string(), 7, 51, vec![])],
            batch_size,
            seq_len,
            d_text,
        );

        println!("\n=== Profiling Inline Batch Generation ===");
        println!("Rayon threads: {}", rayon::current_num_threads());
        println!(
            "Batch size: {}, Seq len: {}, d_text: {}",
            batch_size, seq_len, d_text
        );

        // Warm up
        let _ = sampler.batch(0);

        // Profile multiple runs - now it's all in one unified batch() call
        let num_runs = 10;
        let mut batch_times = Vec::new();

        for run in 0..num_runs {
            let start = Instant::now();
            let _batch = sampler.batch(run);
            batch_times.push(start.elapsed());
        }

        // Calculate averages (skip first run as warmup)
        let avg_time: f64 = batch_times
            .iter()
            .skip(1)
            .map(|d| d.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / (num_runs - 1) as f64;
        let min_time = batch_times.iter().skip(1).min().unwrap().as_secs_f64() * 1000.0;
        let max_time = batch_times.iter().skip(1).max().unwrap().as_secs_f64() * 1000.0;

        println!(
            "\n=== Inline Batch Timing (avg of {} runs, excluding warmup) ===",
            num_runs - 1
        );
        println!(
            "Batch time:        {:7.2} ms (min: {:.2}, max: {:.2})",
            avg_time, min_time, max_time
        );
        println!("Per-sequence:      {:7.2} ms", avg_time / batch_size as f64);
        println!(
            "Sequences/sec:     {:7.0}",
            batch_size as f64 * 1000.0 / avg_time
        );
        println!("Batches/sec:       {:7.1}", 1000.0 / avg_time);

        // Compare buffer reuse vs fresh allocation
        println!("\n=== Buffer Reuse Comparison ===");
        {
            // Test with buffer reuse
            let mut buffer = sampler.create_buffer();

            // Warm up
            sampler.batch_into(&mut buffer, 0);

            let reuse_runs = 10;
            let mut reuse_times = Vec::new();
            for run in 0..reuse_runs {
                let start = Instant::now();
                sampler.batch_into(&mut buffer, run);
                reuse_times.push(start.elapsed());
            }

            let avg_reuse: f64 = reuse_times
                .iter()
                .skip(1)
                .map(|d| d.as_secs_f64() * 1000.0)
                .sum::<f64>()
                / (reuse_runs - 1) as f64;
            let min_reuse = reuse_times.iter().skip(1).min().unwrap().as_secs_f64() * 1000.0;

            println!("Fresh allocation:  {:6.2} ms", avg_time);
            println!(
                "Buffer reuse:      {:6.2} ms (min: {:.2})",
                avg_reuse, min_reuse
            );
            println!("Speedup:           {:6.1}x", avg_time / avg_reuse);
            println!(
                "Time saved:        {:6.2} ms ({:.0}%)",
                avg_time - avg_reuse,
                100.0 * (avg_time - avg_reuse) / avg_time
            );
        }

        // Parallelism analysis
        let estimated_sequential = avg_time * rayon::current_num_threads() as f64;
        println!("\n=== Parallelism Analysis ===");
        println!("Estimated sequential: ~{:.0} ms", estimated_sequential);
        println!("Parallel speedup: {:.1}x", estimated_sequential / avg_time);

        // Memory throughput estimate
        // Data per batch: values + masks
        let bytes_per_batch = (batch_size * seq_len * (4 + 4 + 4 + 4 + 2*5 + 4 + 1 + 1 + 1)) // values/flags
            + (batch_size * seq_len * d_text * 2 * 2) // text embeddings (2 fields, f16=2 bytes)
            + (batch_size * seq_len * seq_len * 3); // attention masks (3 bool masks)
        println!(
            "Data per batch: {:.1} MB",
            bytes_per_batch as f64 / 1024.0 / 1024.0
        );
        println!(
            "Throughput: {:.1} GB/s",
            bytes_per_batch as f64 / avg_time / 1000.0 / 1000.0
        );

        // Detailed instrumentation - time each phase
        println!("\n=== Detailed Phase Timing ===");
        {
            use std::sync::atomic::{AtomicU64, Ordering};

            // Atomic counters for aggregating times across threads (in nanoseconds)
            static ALLOC_TIME: AtomicU64 = AtomicU64::new(0);
            static BFS_TIME: AtomicU64 = AtomicU64::new(0);
            static CELL_FILL_TIME: AtomicU64 = AtomicU64::new(0);
            static TEXT_COPY_TIME: AtomicU64 = AtomicU64::new(0);
            static MASK_TIME: AtomicU64 = AtomicU64::new(0);

            // Reset counters
            ALLOC_TIME.store(0, Ordering::SeqCst);
            BFS_TIME.store(0, Ordering::SeqCst);
            CELL_FILL_TIME.store(0, Ordering::SeqCst);
            TEXT_COPY_TIME.store(0, Ordering::SeqCst);
            MASK_TIME.store(0, Ordering::SeqCst);

            // Run one batch with instrumentation
            let instrumented_start = Instant::now();

            // Pre-allocate
            let alloc_start = Instant::now();
            let mut vecs = BatchVecs::new_preallocated(batch_size, seq_len, d_text, batch_size);
            let alloc_time = alloc_start.elapsed();
            ALLOC_TIME.fetch_add(alloc_time.as_nanos() as u64, Ordering::Relaxed);

            // Get pointers
            let node_ptr = SyncPtr::new(vecs.node_indices.as_mut_ptr());
            let f2p_ptr = SyncPtr::new(vecs.f2p_neighbor_indices.as_mut_ptr());
            let table_ptr = SyncPtr::new(vecs.table_name_indices.as_mut_ptr());
            let col_ptr = SyncPtr::new(vecs.column_name_indices.as_mut_ptr());
            let num_ptr = SyncPtr::new(vecs.number_values.as_mut_ptr());
            let dt_ptr = SyncPtr::new(vecs.datetime_values.as_mut_ptr());
            let bool_ptr = SyncPtr::new(vecs.boolean_values.as_mut_ptr());
            let text_ptr = SyncPtr::new(vecs.text_values.as_mut_ptr());
            let colname_ptr = SyncPtr::new(vecs.column_name_values.as_mut_ptr());
            let sem_ptr = SyncPtr::new(vecs.semantic_types.as_mut_ptr());
            let mask_ptr = SyncPtr::new(vecs.masks.as_mut_ptr());
            let task_ptr = SyncPtr::new(vecs.is_task_node.as_mut_ptr());
            let pad_ptr = SyncPtr::new(vecs.is_padding.as_mut_ptr());
            let col_attn_ptr = SyncPtr::new(vecs.column_attn_mask.as_mut_ptr());
            let feat_attn_ptr = SyncPtr::new(vecs.feature_attn_mask.as_mut_ptr());
            let nbr_attn_ptr = SyncPtr::new(vecs.neighbor_attn_mask.as_mut_ptr());

            let items_len = sampler.items.len();

            // Parallel fill with instrumentation
            (0..batch_size).into_par_iter().for_each(|i| {
                let j = i % items_len;
                let item = &sampler.items[j];
                let db = &sampler.databases[item.database_idx];

                let seq_offset = i * seq_len;
                let f2p_offset = i * seq_len * MAX_F2P_NEIGHBORS;
                let text_offset = i * seq_len * d_text;
                let mask_offset = i * seq_len * seq_len;

                unsafe {
                    let mut seq = SequenceSlice {
                        node_indices: std::slice::from_raw_parts_mut(
                            node_ptr.add(seq_offset),
                            seq_len,
                        ),
                        f2p_neighbor_indices: std::slice::from_raw_parts_mut(
                            f2p_ptr.add(f2p_offset),
                            seq_len * MAX_F2P_NEIGHBORS,
                        ),
                        table_name_indices: std::slice::from_raw_parts_mut(
                            table_ptr.add(seq_offset),
                            seq_len,
                        ),
                        column_name_indices: std::slice::from_raw_parts_mut(
                            col_ptr.add(seq_offset),
                            seq_len,
                        ),
                        number_values: std::slice::from_raw_parts_mut(
                            num_ptr.add(seq_offset),
                            seq_len,
                        ),
                        datetime_values: std::slice::from_raw_parts_mut(
                            dt_ptr.add(seq_offset),
                            seq_len,
                        ),
                        boolean_values: std::slice::from_raw_parts_mut(
                            bool_ptr.add(seq_offset),
                            seq_len,
                        ),
                        text_values: std::slice::from_raw_parts_mut(
                            text_ptr.add(text_offset),
                            seq_len * d_text,
                        ),
                        column_name_values: std::slice::from_raw_parts_mut(
                            colname_ptr.add(text_offset),
                            seq_len * d_text,
                        ),
                        semantic_types: std::slice::from_raw_parts_mut(
                            sem_ptr.add(seq_offset),
                            seq_len,
                        ),
                        masks: std::slice::from_raw_parts_mut(mask_ptr.add(seq_offset), seq_len),
                        is_task_node: std::slice::from_raw_parts_mut(
                            task_ptr.add(seq_offset),
                            seq_len,
                        ),
                        is_padding: std::slice::from_raw_parts_mut(
                            pad_ptr.add(seq_offset),
                            seq_len,
                        ),
                        column_attn_mask: std::slice::from_raw_parts_mut(
                            col_attn_ptr.add(mask_offset),
                            seq_len * seq_len,
                        ),
                        feature_attn_mask: std::slice::from_raw_parts_mut(
                            feat_attn_ptr.add(mask_offset),
                            seq_len * seq_len,
                        ),
                        neighbor_attn_mask: std::slice::from_raw_parts_mut(
                            nbr_attn_ptr.add(mask_offset),
                            seq_len * seq_len,
                        ),
                    };

                    // BFS traversal
                    let bfs_start = Instant::now();
                    let seed_row = db.get_row(item.row_idx);
                    let seed_table = db.get_table(seed_row.table_idx);
                    let seed_timestamp = sampler.get_row_timestamp(db, item.row_idx);
                    let mut visited = vec![false; db.num_rows()];
                    let mut f2p_frontier: Vec<(usize, RowIdx)> = vec![(0, item.row_idx)];
                    let mut p2f_frontier: Vec<Vec<RowIdx>> = Vec::new();
                    let mut seq_i = 0;
                    let mut rng = StdRng::seed_from_u64(item.row_idx.0 as u64);

                    let mut cell_fill_ns = 0u64;
                    let mut text_copy_ns = 0u64;

                    'bfs: loop {
                        let (depth, row_idx) = if let Some(fi) = f2p_frontier.pop() {
                            fi
                        } else {
                            let mut found = None;
                            for (d, nodes) in p2f_frontier.iter().enumerate() {
                                if !nodes.is_empty() {
                                    found = Some(d);
                                    break;
                                }
                            }
                            if let Some(d) = found {
                                let r = rng.random_range(0..p2f_frontier[d].len());
                                let len = p2f_frontier[d].len();
                                p2f_frontier[d].swap(r, len - 1);
                                (d, p2f_frontier[d].pop().unwrap())
                            } else {
                                break 'bfs;
                            }
                        };

                        if visited[row_idx.0 as usize] {
                            continue;
                        }
                        visited[row_idx.0 as usize] = true;

                        let row = db.get_row(row_idx);
                        let table = db.get_table(row.table_idx);
                        let mut f2p_neighbors: Vec<RowIdx> = Vec::new();

                        for &edge_idx in &db.edges_from[row_idx.0 as usize] {
                            let edge = &db.fk_edges[edge_idx];
                            f2p_neighbors.push(edge.to_row);
                            f2p_frontier.push((depth + 1, edge.to_row));
                        }

                        for &edge_idx in &db.edges_to[row_idx.0 as usize] {
                            let edge = &db.fk_edges[edge_idx];
                            if let Some(cutoff) = seed_timestamp {
                                if !db.row_is_before(edge.from_row, cutoff) {
                                    continue;
                                }
                            }
                            let child_table = db.get_table(db.get_row(edge.from_row).table_idx);
                            if child_table.idx == seed_table.idx || item.is_task_table {
                                while p2f_frontier.len() <= depth + 1 {
                                    p2f_frontier.push(Vec::new());
                                }
                                p2f_frontier[depth + 1].push(edge.from_row);
                            }
                        }

                        let col_start = table.column_range.0.0;
                        let is_seed = row_idx == item.row_idx;

                        for (local_idx, cell) in row.normalized.iter().enumerate() {
                            if matches!(cell, NormalizedCellValue::Null) {
                                continue;
                            }
                            let col_idx = ColumnIdx(col_start + local_idx as u32);
                            if is_seed && item.columns_to_drop.contains(&col_idx) {
                                continue;
                            }

                            let cell_start = Instant::now();
                            let col = db.get_column(col_idx);
                            seq.node_indices[seq_i] = row_idx.0 as i32;
                            seq.table_name_indices[seq_i] = row.table_idx.0 as i32;
                            seq.column_name_indices[seq_i] = col_idx.0 as i32;
                            for (j, &nr) in f2p_neighbors.iter().take(MAX_F2P_NEIGHBORS).enumerate()
                            {
                                seq.f2p_neighbor_indices[seq_i * MAX_F2P_NEIGHBORS + j] =
                                    nr.0 as i32;
                            }
                            seq.semantic_types[seq_i] = col.dtype as i32;
                            match (cell, col.dtype) {
                                (NormalizedCellValue::Scalar(v), SemanticType::Number) => {
                                    seq.number_values[seq_i] = f16::from_f32(*v)
                                }
                                (NormalizedCellValue::Scalar(v), SemanticType::Datetime) => {
                                    seq.datetime_values[seq_i] = f16::from_f32(*v)
                                }
                                (NormalizedCellValue::Scalar(v), SemanticType::Boolean) => {
                                    seq.boolean_values[seq_i] = f16::from_f32(*v)
                                }
                                (NormalizedCellValue::Text(ti), SemanticType::Text) => {
                                    let text_start = Instant::now();
                                    if let Some(emb) = db.text_value_embeddings.get(ti.0 as usize) {
                                        let s = seq_i * d_text;
                                        let e = s + d_text.min(emb.len());
                                        seq.text_values[s..e].copy_from_slice(&emb[..e - s]);
                                    }
                                    text_copy_ns += text_start.elapsed().as_nanos() as u64;
                                }
                                _ => {}
                            }
                            if let Some(ref emb) = col.column_description_embedding {
                                let text_start = Instant::now();
                                let s = seq_i * d_text;
                                let e = s + d_text.min(emb.len());
                                seq.column_name_values[s..e].copy_from_slice(&emb[..e - s]);
                                text_copy_ns += text_start.elapsed().as_nanos() as u64;
                            }
                            seq.masks[seq_i] = is_seed && col_idx == item.target_column;
                            seq.is_task_node[seq_i] =
                                item.is_task_table && row.table_idx == seed_row.table_idx;
                            seq.is_padding[seq_i] = false;
                            cell_fill_ns += cell_start.elapsed().as_nanos() as u64;

                            seq_i += 1;
                            if seq_i >= seq_len {
                                break 'bfs;
                            }
                        }
                    }
                    let bfs_time = bfs_start.elapsed();
                    BFS_TIME
                        .fetch_add(bfs_time.as_nanos() as u64 - cell_fill_ns, Ordering::Relaxed);
                    CELL_FILL_TIME.fetch_add(cell_fill_ns - text_copy_ns, Ordering::Relaxed);
                    TEXT_COPY_TIME.fetch_add(text_copy_ns, Ordering::Relaxed);

                    // Attention mask computation
                    let mask_start = Instant::now();
                    sampler.compute_sequence_attention_masks(&mut seq);
                    MASK_TIME.fetch_add(mask_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
            });

            let instrumented_total = instrumented_start.elapsed();

            // Report times (divide by thread count to get wall-clock equivalent)
            let threads = rayon::current_num_threads() as f64;
            let alloc_ms = ALLOC_TIME.load(Ordering::SeqCst) as f64 / 1_000_000.0;
            let bfs_ms = BFS_TIME.load(Ordering::SeqCst) as f64 / 1_000_000.0 / threads;
            let cell_ms = CELL_FILL_TIME.load(Ordering::SeqCst) as f64 / 1_000_000.0 / threads;
            let text_ms = TEXT_COPY_TIME.load(Ordering::SeqCst) as f64 / 1_000_000.0 / threads;
            let mask_ms = MASK_TIME.load(Ordering::SeqCst) as f64 / 1_000_000.0 / threads;
            let total_ms = instrumented_total.as_secs_f64() * 1000.0;
            let other_ms = total_ms - alloc_ms - bfs_ms - cell_ms - text_ms - mask_ms;

            println!(
                "  Allocation:      {:6.2} ms ({:4.1}%)",
                alloc_ms,
                100.0 * alloc_ms / total_ms
            );
            println!(
                "  BFS traversal:   {:6.2} ms ({:4.1}%)",
                bfs_ms,
                100.0 * bfs_ms / total_ms
            );
            println!(
                "  Cell filling:    {:6.2} ms ({:4.1}%)",
                cell_ms,
                100.0 * cell_ms / total_ms
            );
            println!(
                "  Text embedding:  {:6.2} ms ({:4.1}%)",
                text_ms,
                100.0 * text_ms / total_ms
            );
            println!(
                "  Attention masks: {:6.2} ms ({:4.1}%)",
                mask_ms,
                100.0 * mask_ms / total_ms
            );
            println!(
                "  Other/sync:      {:6.2} ms ({:4.1}%)",
                other_ms,
                100.0 * other_ms / total_ms
            );
            println!("  TOTAL:           {:6.2} ms", total_ms);
        }

        // Hardware bottleneck analysis
        println!("\n=== Hardware Bottleneck Analysis ===");

        // Memory bandwidth (typical DDR4: 25-50 GB/s, DDR5: 50-80 GB/s)
        let measured_throughput = bytes_per_batch as f64 / avg_time / 1000.0 / 1000.0;
        let ddr4_typical = 40.0; // GB/s
        let ddr5_typical = 60.0; // GB/s
        println!("Memory bandwidth utilization:");
        println!(
            "  vs DDR4 (~40 GB/s): {:.0}%",
            100.0 * measured_throughput / ddr4_typical
        );
        println!(
            "  vs DDR5 (~60 GB/s): {:.0}%",
            100.0 * measured_throughput / ddr5_typical
        );

        // Attention mask computation cost
        let mask_comparisons = batch_size * seq_len * seq_len * 3; // 3 masks, each is O(seq_len^2)
        let mask_bytes = mask_comparisons; // 1 byte per bool
        println!("\nAttention mask overhead:");
        println!("  Comparisons: {} M", mask_comparisons / 1_000_000);
        println!(
            "  Mask data: {} MB ({:.0}% of batch)",
            mask_bytes / 1024 / 1024,
            100.0 * mask_bytes as f64 / bytes_per_batch as f64
        );

        // Cache analysis
        let l2_size = 256 * 1024; // 256 KB typical per-core L2
        let l3_size = 32 * 1024 * 1024; // 32 MB typical shared L3
        let per_seq_data = bytes_per_batch / batch_size;
        println!("\nCache analysis:");
        println!(
            "  Per-sequence data: {:.1} MB",
            per_seq_data as f64 / 1024.0 / 1024.0
        );
        println!(
            "  Fits in L2 ({} KB): {}",
            l2_size / 1024,
            if per_seq_data < l2_size {
                "YES"
            } else {
                "NO ❌"
            }
        );
        println!(
            "  Fits in L3 ({} MB): {}",
            l3_size / 1024 / 1024,
            if per_seq_data < l3_size {
                "YES ✓"
            } else {
                "NO"
            }
        );
        println!(
            "  Full batch fits in L3: {}",
            if bytes_per_batch < l3_size {
                "YES"
            } else {
                "NO ❌"
            }
        );

        // SIMD efficiency
        let simd_lanes = 8; // AVX2
        let seq_len_remainder = seq_len % simd_lanes;
        println!("\nSIMD efficiency:");
        println!(
            "  seq_len % 8 = {} ({})",
            seq_len_remainder,
            if seq_len_remainder == 0 {
                "perfect ✓"
            } else {
                "has scalar tail"
            }
        );
        println!("  Comparisons per SIMD op: {}", simd_lanes);
        println!("  SIMD iterations per mask row: {}", seq_len / simd_lanes);

        // BFS random access (estimate based on graph structure)
        println!("\nBFS random access pattern:");
        println!("  Graph traversal is cache-unfriendly (random FK lookups)");
        println!(
            "  Visited array: {} KB per sequence",
            sampler.databases[0].num_rows() / 1024
        );

        // Theoretical limits
        let theoretical_mask_only_time =
            mask_comparisons as f64 / (3.0e9 * simd_lanes as f64) * 1000.0; // ~3 GHz, 8-wide
        println!("\nTheoretical limits:");
        println!(
            "  Pure mask computation @3GHz: {:.2} ms (actual: ~50% of batch time)",
            theoretical_mask_only_time
        );
        println!(
            "  If memory-bound at 40 GB/s: {:.2} ms",
            bytes_per_batch as f64 / 40e9 * 1000.0
        );
        println!(
            "  If memory-bound at 60 GB/s: {:.2} ms",
            bytes_per_batch as f64 / 60e9 * 1000.0
        );

        // Bottleneck identification
        println!("\n=== Likely Bottlenecks ===");
        if measured_throughput > ddr4_typical * 0.7 {
            println!("  → MEMORY BANDWIDTH limited");
        }
        if per_seq_data > l2_size {
            println!("  → L2 CACHE misses (data too large for per-core cache)");
        }
        if bytes_per_batch > l3_size {
            println!("  → L3 CACHE misses (full batch exceeds shared cache)");
        }
        println!("  → BFS random access pattern (graph traversal)");
    }
}
