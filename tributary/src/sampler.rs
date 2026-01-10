//! Batch sampler for relational data.
//!
//! Performs BFS traversal of the FK graph starting from seed rows,
//! producing flat vectors that get reshaped in Python for the model.
//!
//! Key design decisions:
//! - Single database per sampler (load multiple samplers for multi-DB training)
//! - Configurable masking strategies (random, balanced, targeted)
//! - Pre-computed attention masks on CPU to offload work from GPU
//! - Thread-local buffers to avoid allocations in hot path

use std::cell::RefCell;
use std::path::Path;

use fixedbitset::FixedBitSet;
use half::f16;
use numpy::PyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

use crate::types::{ColumnIdx, Database, EmbeddingIdx, RowIdx, SemanticType};
use crate::utility::{TIMESTAMP_DIM, expand_timestamp};

// ============================================================================
// Constants
// ============================================================================

/// Maximum number of foreign-to-primary neighbors tracked per cell.
/// Used for computing feature attention masks.
const MAX_F2P_NEIGHBORS: usize = 5;

/// Default mask rate for random masking (15% like BERT).
const DEFAULT_MASK_RATE: f32 = 0.15;

// ============================================================================
// Masking Strategy
// ============================================================================

/// Strategy for selecting which cells to mask during training.
#[derive(Debug, Clone)]
pub enum MaskingStrategy {
    /// Mask cells randomly with given probability.
    /// Good for pre-training.
    Random { mask_rate: f32 },

    /// Mask specific columns on the seed row.
    /// Good for fine-tuning on a specific prediction task.
    TargetColumns { columns: Vec<ColumnIdx> },

    /// Random masking, but ensure each semantic type is represented.
    /// Useful when data is imbalanced (e.g., 90% numerical cells).
    BalancedRandom { mask_rate: f32 },
}

impl Default for MaskingStrategy {
    fn default() -> Self {
        MaskingStrategy::Random {
            mask_rate: DEFAULT_MASK_RATE,
        }
    }
}

// ============================================================================
// Raw Pointer Wrapper for Parallel Access
// ============================================================================

/// Raw pointer wrapper that implements Send+Sync for parallel mutable access.
/// SAFETY: Only use when guaranteeing non-overlapping access from different threads.
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

// ============================================================================
// Sequence Buffers (per-sequence working memory)
// ============================================================================

/// Mutable slices into one sequence's portion of BatchVecs.
struct SequenceSlice<'a> {
    numerical_values: &'a mut [f32],
    categorical_values: &'a mut [f16],
    text_values: &'a mut [f16],
    timestamp_values: &'a mut [f32],
    column_name_values: &'a mut [f16],
    semantic_types: &'a mut [i32],
    masks: &'a mut [bool],
    is_padding: &'a mut [bool],
    /// Bitpacked attention masks: each row of seq_len bits packed into u64s.
    /// Layout: [row0_word0, row0_word1, ..., row1_word0, ...]
    /// Words per row: ceil(seq_len / 64)
    column_attn_mask: &'a mut [u64],
    feature_attn_mask: &'a mut [u64],
    neighbor_attn_mask: &'a mut [u64],
}

/// Intermediate indices for attention mask computation.
struct SequenceIndices {
    /// Node (row) index for each cell position, -1 if padding.
    node: Vec<i32>,
    /// F2P neighbor indices for each cell (flattened: seq_len * MAX_F2P_NEIGHBORS).
    f2p_neighbors: Vec<i32>,
    /// Table index for each cell position.
    table: Vec<i32>,
    /// Column index for each cell position.
    column: Vec<i32>,
}

impl SequenceIndices {
    fn new(seq_len: usize) -> Self {
        Self {
            node: vec![-1; seq_len],
            f2p_neighbors: vec![-1; seq_len * MAX_F2P_NEIGHBORS],
            table: vec![0; seq_len],
            column: vec![0; seq_len],
        }
    }
}

// ============================================================================
// Traversal Buffers (reused across sequences via thread-local storage)
// ============================================================================

/// Reusable buffers for BFS traversal, avoiding per-sequence allocations.
struct TraversalBuffers {
    /// Visited bitset - 8x smaller than Vec<bool>.
    visited: FixedBitSet,
    /// Foreign-to-primary frontier: (depth, row_idx).
    f2p_frontier: Vec<(usize, RowIdx)>,
    /// Primary-to-foreign frontier by depth level.
    p2f_frontier: Vec<Vec<RowIdx>>,
    /// Temporary buffer for f2p neighbors of current node.
    f2p_neighbors: Vec<RowIdx>,
    /// Temporary buffer for children (p2f neighbors).
    children: Vec<RowIdx>,
}

impl TraversalBuffers {
    fn new(num_rows: usize) -> Self {
        Self {
            visited: FixedBitSet::with_capacity(num_rows),
            f2p_frontier: Vec::with_capacity(1024),
            p2f_frontier: Vec::with_capacity(32),
            f2p_neighbors: Vec::with_capacity(16),
            children: Vec::with_capacity(256),
        }
    }

    fn reset(&mut self, num_rows: usize) {
        if self.visited.len() < num_rows {
            self.visited.grow(num_rows);
        }
        self.visited.clear();
        self.f2p_frontier.clear();
        for level in &mut self.p2f_frontier {
            level.clear();
        }
        self.f2p_neighbors.clear();
        self.children.clear();
    }

    /// Pop next node from frontiers: prioritize f2p (parent) edges, then p2f (child) edges.
    /// Returns None when both frontiers are exhausted.
    fn pop_next(&mut self, rng: &mut StdRng) -> Option<(usize, RowIdx)> {
        // Prefer f2p frontier (direct parent traversal)
        if let Some(item) = self.f2p_frontier.pop() {
            return Some(item);
        }
        // Fall back to p2f frontier (random child from earliest depth level)
        for depth in 0..self.p2f_frontier.len() {
            let level = &mut self.p2f_frontier[depth];
            if !level.is_empty() {
                let idx = rng.random_range(0..level.len());
                let row = level.swap_remove(idx);
                return Some((depth, row));
            }
        }
        None
    }
}

thread_local! {
    static TRAVERSAL_BUFFERS: RefCell<Option<TraversalBuffers>> = const { RefCell::new(None) };
}

fn get_traversal_buffers(num_rows: usize) -> TraversalBuffers {
    TRAVERSAL_BUFFERS.with(|cell| {
        let mut opt = cell.borrow_mut();
        match opt.take() {
            Some(mut buffers) => {
                buffers.reset(num_rows);
                buffers
            }
            None => TraversalBuffers::new(num_rows),
        }
    })
}

fn return_traversal_buffers(buffers: TraversalBuffers) {
    TRAVERSAL_BUFFERS.with(|cell| {
        *cell.borrow_mut() = Some(buffers);
    });
}

// ============================================================================
// Batch Vectors (output format)
// ============================================================================

/// Number of u64 words needed to pack `n` bits.
#[inline]
const fn words_for_bits(n: usize) -> usize {
    (n + 63) / 64
}

/// Flat vectors for batch data. Python reshapes these to (batch_size, seq_len, ...).
pub struct BatchVecs {
    /// Z-score normalized numerical values. Shape: (B * S, 1).
    numerical_values: Vec<f32>,
    /// Pre-computed categorical embeddings. Shape: (B * S, d_text).
    categorical_values: Vec<f16>,
    /// Pre-computed text embeddings. Shape: (B * S, d_text).
    text_values: Vec<f16>,
    /// Expanded timestamp features. Shape: (B * S, TIMESTAMP_DIM).
    timestamp_values: Vec<f32>,
    /// Column name embeddings. Shape: (B * S, d_text).
    column_name_values: Vec<f16>,
    /// Semantic type per cell (0=num, 1=cat, 2=ts, 3=text). Shape: (B * S).
    semantic_types: Vec<i32>,
    /// Mask indicating cells to predict. Shape: (B * S).
    masks: Vec<bool>,
    /// Padding indicator. Shape: (B * S).
    is_padding: Vec<bool>,
    /// Bitpacked column attention mask. Shape: (B * S * words_per_row) where words_per_row = ceil(S/64).
    column_attn_mask: Vec<u64>,
    /// Bitpacked feature attention mask. Shape: (B * S * words_per_row).
    feature_attn_mask: Vec<u64>,
    /// Bitpacked neighbor attention mask. Shape: (B * S * words_per_row).
    neighbor_attn_mask: Vec<u64>,
    /// Sequence length (needed for unpacking).
    seq_len: usize,
}

impl BatchVecs {
    /// Create with zeroed memory.
    fn new(batch_size: usize, seq_len: usize, d_text: usize) -> Self {
        let l = batch_size * seq_len;
        // Bitpacked: each row needs ceil(seq_len/64) u64 words
        let words_per_row = words_for_bits(seq_len);
        let packed_size = batch_size * seq_len * words_per_row;

        Self {
            numerical_values: vec![0.0; l],
            categorical_values: vec![f16::ZERO; l * d_text],
            text_values: vec![f16::ZERO; l * d_text],
            timestamp_values: vec![0.0; l * TIMESTAMP_DIM],
            column_name_values: vec![f16::ZERO; l * d_text],
            semantic_types: vec![0; l],
            masks: vec![false; l],
            is_padding: vec![true; l], // Default to padding
            column_attn_mask: vec![0u64; packed_size],
            feature_attn_mask: vec![0u64; packed_size],
            neighbor_attn_mask: vec![0u64; packed_size],
            seq_len,
        }
    }

    /// Reset all fields to their default state for reuse.
    /// This avoids reallocation - just memsets existing memory.
    /// Note: Attention masks are zeroed per-sample in parallel during fill_batch_vecs.
    fn reset(&mut self) {
        // Only is_padding and masks need serial reset
        // Attention masks are handled per-sample in compute_attention_masks
        self.is_padding.fill(true);
        self.masks.fill(false);
    }

    fn into_pyobject(self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        // Unpack bitpacked masks to bool for Python compatibility
        let batch_size = self.is_padding.len() / self.seq_len;
        let unpacked_size = batch_size * self.seq_len * self.seq_len;

        let unpack_mask = |packed: &[u64]| -> Vec<bool> {
            let words_per_row = words_for_bits(self.seq_len);
            let mut unpacked = vec![false; unpacked_size];
            for sample in 0..batch_size {
                for row in 0..self.seq_len {
                    let row_start = (sample * self.seq_len + row) * words_per_row;
                    let out_start = (sample * self.seq_len + row) * self.seq_len;
                    for col in 0..self.seq_len {
                        let word_idx = col / 64;
                        let bit_idx = col % 64;
                        unpacked[out_start + col] =
                            (packed[row_start + word_idx] >> bit_idx) & 1 != 0;
                    }
                }
            }
            unpacked
        };

        let col_mask = unpack_mask(&self.column_attn_mask);
        let feat_mask = unpack_mask(&self.feature_attn_mask);
        let nbr_mask = unpack_mask(&self.neighbor_attn_mask);

        Ok(vec![
            (
                "numerical_values",
                PyArray1::from_vec(py, self.numerical_values),
            )
                .into_py_any(py)?,
            (
                "categorical_values",
                PyArray1::from_vec(py, self.categorical_values),
            )
                .into_py_any(py)?,
            ("text_values", PyArray1::from_vec(py, self.text_values)).into_py_any(py)?,
            (
                "timestamp_values",
                PyArray1::from_vec(py, self.timestamp_values),
            )
                .into_py_any(py)?,
            (
                "column_name_values",
                PyArray1::from_vec(py, self.column_name_values),
            )
                .into_py_any(py)?,
            (
                "semantic_types",
                PyArray1::from_vec(py, self.semantic_types),
            )
                .into_py_any(py)?,
            ("masks", PyArray1::from_vec(py, self.masks)).into_py_any(py)?,
            ("is_padding", PyArray1::from_vec(py, self.is_padding)).into_py_any(py)?,
            ("column_attn_mask", PyArray1::from_vec(py, col_mask)).into_py_any(py)?,
            ("feature_attn_mask", PyArray1::from_vec(py, feat_mask)).into_py_any(py)?,
            ("neighbor_attn_mask", PyArray1::from_vec(py, nbr_mask)).into_py_any(py)?,
        ])
    }
}

// ============================================================================
// Sampler Configuration
// ============================================================================

/// Configuration for the sampler.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub max_bfs_width: usize,
    pub masking_strategy: MaskingStrategy,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            seq_len: 1024,
            max_bfs_width: 256,
            masking_strategy: MaskingStrategy::default(),
            seed: 42,
        }
    }
}

// ============================================================================
// Sampler
// ============================================================================

/// Graph sampler for relational databases.
///
/// Loads a preprocessed database and samples BFS neighborhoods for training.
#[pyclass]
pub struct Sampler {
    database: Database,
    /// Seed rows for BFS traversal (shuffled each epoch).
    seeds: Vec<RowIdx>,
    config: SamplerConfig,
    d_text: usize,
    epoch: u64,
}

#[pymethods]
impl Sampler {
    /// Create a new Sampler from a preprocessed database file.
    ///
    /// Args:
    ///     db_path: Path to the .rkyv database file
    ///     batch_size: Number of sequences per batch
    ///     seq_len: Maximum sequence length (cells per sequence)
    ///     max_bfs_width: Max neighbors sampled per BFS step
    ///     mask_rate: Probability of masking each cell (for random masking)
    ///     seed: Random seed for reproducibility
    #[new]
    #[pyo3(signature = (db_path, batch_size=32, seq_len=1024, max_bfs_width=256, mask_rate=0.15, seed=42))]
    fn new(
        db_path: String,
        batch_size: usize,
        seq_len: usize,
        max_bfs_width: usize,
        mask_rate: f32,
        seed: u64,
    ) -> PyResult<Self> {
        let database = Database::load(Path::new(&db_path)).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load database: {}", e))
        })?;

        let config = SamplerConfig {
            batch_size,
            seq_len,
            max_bfs_width,
            masking_strategy: MaskingStrategy::Random { mask_rate },
            seed,
        };

        Ok(Self::init(database, config))
    }

    /// Number of batches in one epoch.
    fn len_py(&self) -> usize {
        self.seeds.len().div_ceil(self.config.batch_size)
    }

    /// Get a batch by index.
    fn batch_py(&self, py: Python<'_>, batch_idx: usize) -> PyResult<Vec<Py<PyAny>>> {
        self.batch(batch_idx).into_pyobject(py)
    }

    /// Shuffle seeds for a new epoch.
    fn shuffle_py(&mut self, epoch: u64) {
        self.epoch = epoch;
        let mut rng = StdRng::seed_from_u64(epoch.wrapping_add(self.config.seed));
        self.seeds.shuffle(&mut rng);
    }

    /// Number of rows in the database.
    fn num_rows(&self) -> usize {
        self.database.num_rows()
    }

    /// Number of tables in the database.
    fn num_tables(&self) -> usize {
        self.database.num_tables()
    }

    /// Embedding dimension.
    fn embed_dim(&self) -> usize {
        self.d_text
    }
}

impl Sampler {
    /// Generate a batch of sequences.
    fn batch(&self, batch_idx: usize) -> BatchVecs {
        let start_idx = batch_idx * self.config.batch_size;
        let actual_batch_size = self
            .config
            .batch_size
            .min(self.seeds.len().saturating_sub(start_idx));

        let mut vecs = BatchVecs::new(self.config.batch_size, self.config.seq_len, self.d_text);

        if actual_batch_size == 0 {
            return vecs;
        }

        self.fill_batch_vecs(&mut vecs, start_idx, actual_batch_size);
        vecs
    }

    /// Fill batch vectors in parallel.
    fn fill_batch_vecs(&self, vecs: &mut BatchVecs, start_idx: usize, actual_batch_size: usize) {
        let seq_len = self.config.seq_len;
        let d_text = self.d_text;
        let words_per_row = words_for_bits(seq_len);

        // Create sync pointers for parallel access
        let num_ptr = SyncPtr::new(vecs.numerical_values.as_mut_ptr());
        let cat_ptr = SyncPtr::new(vecs.categorical_values.as_mut_ptr());
        let text_ptr = SyncPtr::new(vecs.text_values.as_mut_ptr());
        let ts_ptr = SyncPtr::new(vecs.timestamp_values.as_mut_ptr());
        let colname_ptr = SyncPtr::new(vecs.column_name_values.as_mut_ptr());
        let sem_ptr = SyncPtr::new(vecs.semantic_types.as_mut_ptr());
        let mask_ptr = SyncPtr::new(vecs.masks.as_mut_ptr());
        let pad_ptr = SyncPtr::new(vecs.is_padding.as_mut_ptr());
        let col_attn_ptr = SyncPtr::new(vecs.column_attn_mask.as_mut_ptr());
        let feat_attn_ptr = SyncPtr::new(vecs.feature_attn_mask.as_mut_ptr());
        let nbr_attn_ptr = SyncPtr::new(vecs.neighbor_attn_mask.as_mut_ptr());

        (0..actual_batch_size).into_par_iter().for_each(|i| {
            let seed_row = self.seeds[start_idx + i];

            let seq_offset = i * seq_len;
            let text_offset = i * seq_len * d_text;
            let ts_offset = i * seq_len * TIMESTAMP_DIM;
            // Packed mask offset: each sequence has seq_len rows, each row has words_per_row u64s
            let packed_mask_offset = i * seq_len * words_per_row;
            let packed_mask_size = seq_len * words_per_row;

            let mut indices = SequenceIndices::new(seq_len);
            let mut trav = get_traversal_buffers(self.database.num_rows());

            unsafe {
                let slice = SequenceSlice {
                    numerical_values: std::slice::from_raw_parts_mut(
                        num_ptr.add(seq_offset),
                        seq_len,
                    ),
                    categorical_values: std::slice::from_raw_parts_mut(
                        cat_ptr.add(text_offset),
                        seq_len * d_text,
                    ),
                    text_values: std::slice::from_raw_parts_mut(
                        text_ptr.add(text_offset),
                        seq_len * d_text,
                    ),
                    timestamp_values: std::slice::from_raw_parts_mut(
                        ts_ptr.add(ts_offset),
                        seq_len * TIMESTAMP_DIM,
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
                    is_padding: std::slice::from_raw_parts_mut(pad_ptr.add(seq_offset), seq_len),
                    column_attn_mask: std::slice::from_raw_parts_mut(
                        col_attn_ptr.add(packed_mask_offset),
                        packed_mask_size,
                    ),
                    feature_attn_mask: std::slice::from_raw_parts_mut(
                        feat_attn_ptr.add(packed_mask_offset),
                        packed_mask_size,
                    ),
                    neighbor_attn_mask: std::slice::from_raw_parts_mut(
                        nbr_attn_ptr.add(packed_mask_offset),
                        packed_mask_size,
                    ),
                };

                self.fill_sequence(seed_row, slice, &mut indices, &mut trav);
            }

            return_traversal_buffers(trav);
        });
    }

    /// Fill a single sequence via BFS traversal.
    fn fill_sequence(
        &self,
        seed_row: RowIdx,
        mut seq: SequenceSlice<'_>,
        idx: &mut SequenceIndices,
        trav: &mut TraversalBuffers,
    ) {
        // Zero attention masks upfront (done in parallel per-sample)
        // This is faster than iterating through padding rows later
        seq.column_attn_mask.fill(0);
        seq.feature_attn_mask.fill(0);
        seq.neighbor_attn_mask.fill(0);

        let db = &self.database;
        let seed_table_idx = db.row_table(seed_row);
        let seed_table = db.table(seed_table_idx);
        let seed_timestamp = self.get_row_timestamp(seed_row);

        // Initialize BFS with seed row
        trav.f2p_frontier.push((0, seed_row));

        let mut seq_i = 0;
        let mut rng = StdRng::seed_from_u64(
            self.epoch
                .wrapping_add(seed_row.0 as u64)
                .wrapping_add(self.config.seed),
        );

        // BFS traversal
        while let Some((depth, row_idx)) = trav.pop_next(&mut rng) {
            if trav.visited.contains(row_idx.0 as usize) {
                continue;
            }
            trav.visited.insert(row_idx.0 as usize);

            let table_idx = db.row_table(row_idx);

            // Collect f2p neighbors (rows this row points TO via FK)
            trav.f2p_neighbors.clear();
            for &neighbor in db.outgoing_neighbors(row_idx) {
                let neighbor_row = RowIdx(neighbor);
                trav.f2p_neighbors.push(neighbor_row);
                trav.f2p_frontier.push((depth + 1, neighbor_row));
            }

            // Collect p2f neighbors (rows that point TO this row via FK)
            trav.children.clear();
            for &neighbor in db.incoming_neighbors(row_idx) {
                let child_row = RowIdx(neighbor);

                // Temporal constraint: don't include future rows
                if let Some(cutoff) = seed_timestamp {
                    if let Some(child_ts) = self.get_row_timestamp(child_row) {
                        if child_ts > cutoff {
                            continue;
                        }
                    }
                }

                // Only include children from the seed table (for task-focused sampling)
                // or all children (for broader exploration)
                let child_table_idx = db.row_table(child_row);
                if child_table_idx == seed_table.idx {
                    trav.children.push(child_row);
                }
            }

            // Subsample if too many children
            let num_children = trav.children.len();
            let sample_count = num_children.min(self.config.max_bfs_width);

            if num_children > self.config.max_bfs_width {
                for i in 0..sample_count {
                    let j = rng.random_range(i..num_children);
                    trav.children.swap(i, j);
                }
            }

            // Add sampled children to p2f frontier
            for i in 0..sample_count {
                let child_row = trav.children[i];
                while trav.p2f_frontier.len() <= depth + 1 {
                    trav.p2f_frontier.push(Vec::with_capacity(64));
                }
                trav.p2f_frontier[depth + 1].push(child_row);
            }

            // Fill cells from this row
            let row_cells = db.row_cells(row_idx);
            let table_columns = db.table_columns(table_idx);
            let is_seed_row = row_idx == seed_row;

            for (local_idx, &packed_cell) in row_cells.iter().enumerate() {
                // Skip null cells
                if crate::types::is_packed_null(packed_cell) {
                    continue;
                }

                let column = &table_columns[local_idx];
                let global_col_idx = column.idx;

                // Fill indices for attention mask computation
                idx.node[seq_i] = row_idx.0 as i32;
                idx.table[seq_i] = table_idx.0 as i32;
                idx.column[seq_i] = global_col_idx.0 as i32;

                for (j, &neighbor_row) in trav
                    .f2p_neighbors
                    .iter()
                    .take(MAX_F2P_NEIGHBORS)
                    .enumerate()
                {
                    idx.f2p_neighbors[seq_i * MAX_F2P_NEIGHBORS + j] = neighbor_row.0 as i32;
                }

                // Fill semantic type
                seq.semantic_types[seq_i] = column.stype as i32;

                // Fill cell value based on semantic type
                self.fill_cell_value(&mut seq, seq_i, packed_cell, column.stype);

                // Fill column embedding
                let col_embedding = db.get_column_embedding(global_col_idx);
                let start = seq_i * self.d_text;
                let end = start + self.d_text;
                seq.column_name_values[start..end].copy_from_slice(col_embedding);

                // Apply masking strategy
                seq.masks[seq_i] = self.should_mask(&mut rng, is_seed_row, global_col_idx);
                seq.is_padding[seq_i] = false;

                seq_i += 1;
                if seq_i >= self.config.seq_len {
                    break;
                }
            }

            if seq_i >= self.config.seq_len {
                break;
            }
        }

        // Compute attention masks
        self.compute_attention_masks(&mut seq, idx);
    }

    /// Fill cell value into the appropriate buffer based on semantic type.
    fn fill_cell_value(
        &self,
        seq: &mut SequenceSlice<'_>,
        seq_i: usize,
        packed_cell: u32,
        stype: SemanticType,
    ) {
        let db = &self.database;

        match stype {
            SemanticType::Numerical => {
                // Packed cell is f32 bits (already z-scored)
                let value = f32::from_bits(packed_cell);
                seq.numerical_values[seq_i] = value;
            }
            SemanticType::Categorical => {
                // Packed cell is embedding index
                let emb_idx = EmbeddingIdx(packed_cell);
                let embedding = db.get_embedding(emb_idx);
                let start = seq_i * self.d_text;
                let end = start + self.d_text;
                seq.categorical_values[start..end].copy_from_slice(embedding);
            }
            SemanticType::Timestamp => {
                // Packed cell is epoch seconds as f32 bits
                let epoch_secs = f32::from_bits(packed_cell);
                let mean = db.timestamp_mean.unwrap_or(0.0);
                let std = db.timestamp_std.unwrap_or(1.0);
                let expanded = expand_timestamp(epoch_secs, mean, std);
                let start = seq_i * TIMESTAMP_DIM;
                seq.timestamp_values[start..start + TIMESTAMP_DIM].copy_from_slice(&expanded);
            }
            SemanticType::Text => {
                // Packed cell is embedding index
                let emb_idx = EmbeddingIdx(packed_cell);
                let embedding = db.get_embedding(emb_idx);
                let start = seq_i * self.d_text;
                let end = start + self.d_text;
                seq.text_values[start..end].copy_from_slice(embedding);
            }
        }
    }

    /// Determine if a cell should be masked based on the masking strategy.
    fn should_mask(&self, rng: &mut StdRng, is_seed_row: bool, col_idx: ColumnIdx) -> bool {
        match &self.config.masking_strategy {
            MaskingStrategy::Random { mask_rate } => rng.random::<f32>() < *mask_rate,
            MaskingStrategy::TargetColumns { columns } => is_seed_row && columns.contains(&col_idx),
            MaskingStrategy::BalancedRandom { mask_rate } => {
                // TODO: Track type counts and balance masking
                // For now, fall back to random
                rng.random::<f32>() < *mask_rate
            }
        }
    }

    /// Get the timestamp of a row (if it has a time column).
    fn get_row_timestamp(&self, row_idx: RowIdx) -> Option<f32> {
        let db = &self.database;
        let table_idx = db.row_table(row_idx);
        let table = db.table(table_idx);

        let time_col_idx = table.time_column?;

        // Get the local index of the time column within this row
        let local_idx = (time_col_idx.0 - table.column_range.0.0) as usize;

        let row_cells = db.row_cells(row_idx);
        if local_idx >= row_cells.len() {
            return None;
        }

        let packed_cell = row_cells[local_idx];
        if crate::types::is_packed_null(packed_cell) {
            return None;
        }

        // Timestamps are stored as epoch seconds (f32 bits)
        Some(f32::from_bits(packed_cell))
    }

    /// Compute attention masks for the sequence with bitpacking (SIMD-optimized for x86_64).
    #[cfg(target_arch = "x86_64")]
    fn compute_attention_masks(&self, seq: &mut SequenceSlice<'_>, idx: &SequenceIndices) {
        use std::arch::x86_64::*;

        let seq_len = self.config.seq_len;
        let words_per_row = words_for_bits(seq_len);

        unsafe {
            let minus_one = _mm256_set1_epi32(-1);

            for q in 0..seq_len {
                // Skip padding rows (already zeroed in fill_sequence)
                if seq.is_padding[q] {
                    continue;
                }

                let q_node = idx.node[q];
                let q_table = idx.table[q];
                let q_col = idx.column[q];
                let q_f2p_start = q * MAX_F2P_NEIGHBORS;

                let q_node_v = _mm256_set1_epi32(q_node);
                let q_table_v = _mm256_set1_epi32(q_table);
                let q_col_v = _mm256_set1_epi32(q_col);

                // Pre-broadcast q's f2p neighbors
                let q_f2p: [__m256i; MAX_F2P_NEIGHBORS] =
                    std::array::from_fn(|i| _mm256_set1_epi32(idx.f2p_neighbors[q_f2p_start + i]));

                // Get mutable slices for this row of each mask
                let row_start = q * words_per_row;
                let col_row = &mut seq.column_attn_mask[row_start..row_start + words_per_row];
                let feat_row = &mut seq.feature_attn_mask[row_start..row_start + words_per_row];
                let nbr_row = &mut seq.neighbor_attn_mask[row_start..row_start + words_per_row];

                // Process each u64 word (64 bits = 8 SIMD iterations of 8 elements each)
                for word_idx in 0..words_per_row {
                    let kv_base = word_idx * 64;
                    let mut col_word = 0u64;
                    let mut feat_word = 0u64;
                    let mut nbr_word = 0u64;

                    // Process 8 kv positions at a time with AVX2
                    let mut kv = 0usize;
                    while kv < 64 && kv_base + kv < seq_len {
                        let kv_abs = kv_base + kv;
                        let remaining = (seq_len - kv_abs).min(8);

                        if remaining == 8 {
                            // Full SIMD path: process 8 positions
                            let kv_nodes =
                                _mm256_loadu_si256(idx.node.as_ptr().add(kv_abs) as *const __m256i);
                            let kv_tables = _mm256_loadu_si256(
                                idx.table.as_ptr().add(kv_abs) as *const __m256i
                            );
                            let kv_cols = _mm256_loadu_si256(
                                idx.column.as_ptr().add(kv_abs) as *const __m256i
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

                            // Neighbor attention: q_node in kv's f2p neighbors
                            let kv_f2p_indices = _mm256_set_epi32(
                                ((kv_abs + 7) * MAX_F2P_NEIGHBORS) as i32,
                                ((kv_abs + 6) * MAX_F2P_NEIGHBORS) as i32,
                                ((kv_abs + 5) * MAX_F2P_NEIGHBORS) as i32,
                                ((kv_abs + 4) * MAX_F2P_NEIGHBORS) as i32,
                                ((kv_abs + 3) * MAX_F2P_NEIGHBORS) as i32,
                                ((kv_abs + 2) * MAX_F2P_NEIGHBORS) as i32,
                                ((kv_abs + 1) * MAX_F2P_NEIGHBORS) as i32,
                                (kv_abs * MAX_F2P_NEIGHBORS) as i32,
                            );
                            let mut nbr_result = _mm256_setzero_si256();
                            for i in 0..MAX_F2P_NEIGHBORS {
                                let offset_indices =
                                    _mm256_add_epi32(kv_f2p_indices, _mm256_set1_epi32(i as i32));
                                let kv_neighbors = _mm256_i32gather_epi32::<4>(
                                    idx.f2p_neighbors.as_ptr(),
                                    offset_indices,
                                );
                                let matches = _mm256_cmpeq_epi32(kv_neighbors, q_node_v);
                                nbr_result = _mm256_or_si256(nbr_result, matches);
                            }

                            // Extract 8 bits from each result
                            // movemask gives us 32 bits (4 bits per lane), we need 1 bit per lane
                            let col_mask =
                                _mm256_movemask_ps(_mm256_castsi256_ps(col_result)) as u8;
                            let feat_mask =
                                _mm256_movemask_ps(_mm256_castsi256_ps(feat_result)) as u8;
                            let nbr_mask =
                                _mm256_movemask_ps(_mm256_castsi256_ps(nbr_result)) as u8;

                            // Apply padding mask
                            let mut pad_mask = 0xFFu8;
                            for lane in 0..8 {
                                if seq.is_padding[kv_abs + lane] {
                                    pad_mask &= !(1u8 << lane);
                                }
                            }

                            col_word |= ((col_mask & pad_mask) as u64) << kv;
                            feat_word |= ((feat_mask & pad_mask) as u64) << kv;
                            nbr_word |= ((nbr_mask & pad_mask) as u64) << kv;

                            kv += 8;
                        } else {
                            // Scalar fallback for remainder
                            for lane in 0..remaining {
                                let kv_pos = kv_abs + lane;
                                if seq.is_padding[kv_pos] {
                                    continue;
                                }

                                let kv_node = idx.node[kv_pos];
                                let kv_table = idx.table[kv_pos];
                                let kv_col = idx.column[kv_pos];
                                let bit_pos = kv + lane;

                                if q_col == kv_col && q_table == kv_table {
                                    col_word |= 1u64 << bit_pos;
                                }

                                let same_node = q_node == kv_node;
                                let kv_in_q_f2p = (0..MAX_F2P_NEIGHBORS).any(|i| {
                                    let n = idx.f2p_neighbors[q_f2p_start + i];
                                    n >= 0 && n == kv_node
                                });
                                if same_node || kv_in_q_f2p {
                                    feat_word |= 1u64 << bit_pos;
                                }

                                let kv_f2p_start = kv_pos * MAX_F2P_NEIGHBORS;
                                let q_in_kv_f2p = (0..MAX_F2P_NEIGHBORS)
                                    .any(|i| idx.f2p_neighbors[kv_f2p_start + i] == q_node);
                                if q_in_kv_f2p {
                                    nbr_word |= 1u64 << bit_pos;
                                }
                            }
                            kv += remaining;
                        }
                    }

                    col_row[word_idx] = col_word;
                    feat_row[word_idx] = feat_word;
                    nbr_row[word_idx] = nbr_word;
                }
            }
        }
    }

    /// Compute attention masks for the sequence with bitpacking (scalar fallback).
    #[cfg(not(target_arch = "x86_64"))]
    fn compute_attention_masks(&self, seq: &mut SequenceSlice<'_>, idx: &SequenceIndices) {
        let seq_len = self.config.seq_len;
        let words_per_row = words_for_bits(seq_len);

        for q in 0..seq_len {
            // Skip padding rows (already zeroed in fill_sequence)
            if seq.is_padding[q] {
                continue;
            }

            let q_node = idx.node[q];
            let q_table = idx.table[q];
            let q_col = idx.column[q];
            let q_f2p_start = q * MAX_F2P_NEIGHBORS;

            let row_start = q * words_per_row;
            let col_row = &mut seq.column_attn_mask[row_start..row_start + words_per_row];
            let feat_row = &mut seq.feature_attn_mask[row_start..row_start + words_per_row];
            let nbr_row = &mut seq.neighbor_attn_mask[row_start..row_start + words_per_row];

            for word_idx in 0..words_per_row {
                let kv_start = word_idx * 64;
                let kv_end = (kv_start + 64).min(seq_len);

                let mut col_word = 0u64;
                let mut feat_word = 0u64;
                let mut nbr_word = 0u64;

                for kv in kv_start..kv_end {
                    if seq.is_padding[kv] {
                        continue;
                    }

                    let kv_node = idx.node[kv];
                    let kv_table = idx.table[kv];
                    let kv_col = idx.column[kv];
                    let bit_pos = kv - kv_start;

                    if q_col == kv_col && q_table == kv_table {
                        col_word |= 1u64 << bit_pos;
                    }

                    let same_node = q_node == kv_node;
                    let kv_in_q_f2p = (0..MAX_F2P_NEIGHBORS).any(|i| {
                        let n = idx.f2p_neighbors[q_f2p_start + i];
                        n >= 0 && n == kv_node
                    });
                    if same_node || kv_in_q_f2p {
                        feat_word |= 1u64 << bit_pos;
                    }

                    let kv_f2p_start = kv * MAX_F2P_NEIGHBORS;
                    let q_in_kv_f2p = (0..MAX_F2P_NEIGHBORS)
                        .any(|i| idx.f2p_neighbors[kv_f2p_start + i] == q_node);
                    if q_in_kv_f2p {
                        nbr_word |= 1u64 << bit_pos;
                    }
                }

                col_row[word_idx] = col_word;
                feat_row[word_idx] = feat_word;
                nbr_row[word_idx] = nbr_word;
            }
        }
    }
}

// ============================================================================
// Non-PyO3 Constructor (for testing)
// ============================================================================

impl Sampler {
    /// Common initialization logic for both PyO3 and Rust-native constructors.
    fn init(database: Database, config: SamplerConfig) -> Self {
        let d_text = database.embed_dim as usize;
        let seeds: Vec<RowIdx> = (0..database.num_rows()).map(|i| RowIdx(i as u32)).collect();
        Self {
            database,
            seeds,
            config,
            d_text,
            epoch: 0,
        }
    }

    /// Create a new Sampler from a path, without requiring PyO3.
    /// This is the Rust-native version for use in tests and internal code.
    pub fn from_path(db_path: &Path, config: SamplerConfig) -> std::io::Result<Self> {
        let database = Database::load(db_path)?;
        Ok(Self::init(database, config))
    }

    /// Get the database reference for inspection.
    pub fn database(&self) -> &Database {
        &self.database
    }

    /// Get the configuration.
    pub fn config(&self) -> &SamplerConfig {
        &self.config
    }

    /// Number of batches.
    pub fn num_batches(&self) -> usize {
        self.seeds.len().div_ceil(self.config.batch_size)
    }

    /// Generate a batch (public accessor for testing).
    /// NOTE: This allocates a new BatchVecs each call. For better performance,
    /// use `get_batch_reuse` which reuses an internal buffer.
    pub fn get_batch(&self, batch_idx: usize) -> BatchVecs {
        self.batch(batch_idx)
    }

    /// Generate a batch into the provided buffer, reusing memory.
    /// This avoids allocation overhead by reusing the same BatchVecs.
    pub fn fill_batch_into(&self, batch_idx: usize, vecs: &mut BatchVecs) {
        vecs.reset();

        let start_idx = batch_idx * self.config.batch_size;
        let actual_batch_size = self
            .config
            .batch_size
            .min(self.seeds.len().saturating_sub(start_idx));

        if actual_batch_size > 0 {
            self.fill_batch_vecs(vecs, start_idx, actual_batch_size);
        }
    }

    /// Create a reusable BatchVecs buffer sized for this sampler's config.
    pub fn create_batch_buffer(&self) -> BatchVecs {
        BatchVecs::new(self.config.batch_size, self.config.seq_len, self.d_text)
    }

    /// Shuffle seeds for a new epoch (Rust-native version).
    pub fn shuffle(&mut self, epoch: u64) {
        self.epoch = epoch;
        let mut rng = StdRng::seed_from_u64(epoch.wrapping_add(self.config.seed));
        self.seeds.shuffle(&mut rng);
    }
}

/// Generate slice accessor methods for BatchVecs fields.
macro_rules! batch_vec_accessors {
    ($($name:ident: $ty:ty),* $(,)?) => {
        impl BatchVecs {
            $(
                pub fn $name(&self) -> &[$ty] { &self.$name }
            )*

            pub fn seq_len(&self) -> usize { self.semantic_types.len() }
        }
    };
}

batch_vec_accessors! {
    semantic_types: i32,
    numerical_values: f32,
    masks: bool,
    is_padding: bool,
    categorical_values: f16,
    text_values: f16,
    timestamp_values: f32,
    column_name_values: f16,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masking_strategy_default() {
        let strategy = MaskingStrategy::default();
        match strategy {
            MaskingStrategy::Random { mask_rate } => {
                assert!((mask_rate - 0.15).abs() < 0.001);
            }
            _ => panic!("Expected Random masking strategy"),
        }
    }
}
