// Batch sampler for relational data.
// Performs BFS traversal of the FK graph starting from seed rows,
// producing flat vectors that get reshaped in Python for the model.

use crate::types::{
    ColumnIdx, Database, NormalizedCellValue, RawCellValue, RowIdx, SemanticType, TableIdx,
};
use fixedbitset::FixedBitSet;
use half::f16;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::{IntoPyObjectExt, Py, PyAny};
use rand::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;
use std::path::Path;

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

/// Maximum number of foreign-to-primary neighbors tracked per cell.
/// Must match MAX_F2P_NEIGHBORS in model.py
const MAX_F2P_NEIGHBORS: usize = 5;

/// Mutable slices into one sequence's portion of BatchVecs (outputs only).
struct SequenceSlice<'a> {
    number_values: &'a mut [f16],
    datetime_values: &'a mut [f16],
    boolean_values: &'a mut [f16],
    text_values: &'a mut [f16],
    column_name_values: &'a mut [f16],
    semantic_types: &'a mut [i32],
    masks: &'a mut [bool],
    is_task_node: &'a mut [bool],
    is_padding: &'a mut [bool],
    column_attn_mask: &'a mut [bool],
    feature_attn_mask: &'a mut [bool],
    neighbor_attn_mask: &'a mut [bool],
}

/// Intermediate indices for attention mask computation (local to each sequence).
struct SequenceIndices {
    node: Vec<i32>,
    f2p_neighbors: Vec<i32>,
    table: Vec<i32>,
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

/// Reusable buffers for BFS traversal, avoiding per-sequence allocations.
/// These are pooled per-thread and reused across sequences.
struct TraversalBuffers {
    /// Visited bitset - 8x smaller than Vec<bool>
    visited: FixedBitSet,
    /// Foreign-to-primary frontier: (depth, row_idx)
    f2p_frontier: Vec<(usize, RowIdx)>,
    /// Primary-to-foreign frontier by depth level
    p2f_frontier: Vec<Vec<RowIdx>>,
    /// Temporary buffer for f2p neighbors
    f2p_neighbors: Vec<RowIdx>,
    /// Temporary buffer for children
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

    /// Reset all buffers for reuse without deallocating.
    fn reset(&mut self, num_rows: usize) {
        // Grow visited if needed, then clear
        if self.visited.len() < num_rows {
            self.visited.grow(num_rows);
        }
        self.visited.clear();

        // Clear vectors but keep capacity
        self.f2p_frontier.clear();
        for level in &mut self.p2f_frontier {
            level.clear();
        }
        self.f2p_neighbors.clear();
        self.children.clear();
    }
}

// Thread-local storage for traversal buffers to avoid repeated allocations
thread_local! {
    static TRAVERSAL_BUFFERS: RefCell<Option<TraversalBuffers>> = const { RefCell::new(None) };
}

/// Get or create thread-local traversal buffers, resizing if needed.
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

/// Return traversal buffers to thread-local storage for reuse.
fn return_traversal_buffers(buffers: TraversalBuffers) {
    TRAVERSAL_BUFFERS.with(|cell| {
        *cell.borrow_mut() = Some(buffers);
    });
}

/// Flat vectors for batch data. Python reshapes these to (batch_size, seq_len, ...).
struct BatchVecs {
    number_values: Vec<f16>,
    datetime_values: Vec<f16>,
    boolean_values: Vec<f16>,
    text_values: Vec<f16>,
    column_name_values: Vec<f16>,
    semantic_types: Vec<i32>,
    masks: Vec<bool>,
    is_task_node: Vec<bool>,
    is_padding: Vec<bool>,
    column_attn_mask: Vec<bool>,
    feature_attn_mask: Vec<bool>,
    neighbor_attn_mask: Vec<bool>,
    true_batch_size: usize,
}

impl BatchVecs {
    /// Create with uninitialized memory. Caller MUST fill all positions.
    unsafe fn new(batch_size: usize, seq_len: usize, d_text: usize) -> Self {
        let l = batch_size * seq_len;
        let l_sq = batch_size * seq_len * seq_len;

        fn uninit_vec<T>(len: usize) -> Vec<T> {
            let mut v = Vec::with_capacity(len);
            unsafe { v.set_len(len) };
            v
        }

        Self {
            number_values: uninit_vec(l),
            datetime_values: uninit_vec(l),
            boolean_values: uninit_vec(l),
            text_values: uninit_vec(l * d_text),
            column_name_values: uninit_vec(l * d_text),
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

    /// Debug dump of the batch contents
    pub fn dump_debug(&self, seq_len: usize, d_text: usize) {
        println!(
            "╔══════════════════════════════════════════════════════════════════════════════╗"
        );
        println!(
            "║                           BATCH DEBUG DUMP                                   ║"
        );
        println!(
            "╚══════════════════════════════════════════════════════════════════════════════╝"
        );
        println!();

        let batch_size = self.number_values.len() / seq_len;
        let non_padding_count: usize = self.is_padding.iter().filter(|&&p| !p).count();
        let masked_count: usize = self.masks.iter().filter(|&&m| m).count();
        let task_node_count: usize = self.is_task_node.iter().filter(|&&t| t).count();

        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ SUMMARY                                                                     │");
        println!("├─────────────────────────────────────────────────────────────────────────────┤");
        println!(
            "│ Batch size:          {:>10}                                            │",
            batch_size
        );
        println!(
            "│ True batch size:     {:>10}                                            │",
            self.true_batch_size
        );
        println!(
            "│ Seq len:             {:>10}                                            │",
            seq_len
        );
        println!(
            "│ d_text:              {:>10}                                            │",
            d_text
        );
        println!(
            "│ Non-padding cells:   {:>10}                                            │",
            non_padding_count
        );
        println!(
            "│ Masked (target):     {:>10}                                            │",
            masked_count
        );
        println!(
            "│ Task node cells:     {:>10}                                            │",
            task_node_count
        );
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        println!();

        // For each sequence in the batch
        for seq_idx in 0..batch_size {
            let seq_start = seq_idx * seq_len;
            let seq_end = seq_start + seq_len;
            let seq_non_padding: usize = self.is_padding[seq_start..seq_end]
                .iter()
                .filter(|&&p| !p)
                .count();

            println!(
                "┌─────────────────────────────────────────────────────────────────────────────┐"
            );
            println!(
                "│ SEQUENCE {} ({} non-padding cells)                               │",
                seq_idx, seq_non_padding
            );
            println!(
                "└─────────────────────────────────────────────────────────────────────────────┘"
            );

            // Print each non-padding cell
            let mut printed = 0;
            for i in 0..seq_len {
                let idx = seq_start + i;
                if self.is_padding[idx] {
                    continue;
                }

                let sem_type = match self.semantic_types[idx] {
                    0 => "Number",
                    1 => "Text",
                    2 => "Datetime",
                    3 => "Boolean",
                    _ => "Unknown",
                };

                let value_str = match self.semantic_types[idx] {
                    0 => format!("{:.4}", self.number_values[idx]),
                    1 => {
                        // Show first few embedding dims
                        let text_start = idx * d_text;
                        let text_end = (text_start + 4).min(text_start + d_text);
                        let dims: Vec<String> = self.text_values[text_start..text_end]
                            .iter()
                            .map(|v| format!("{:.2}", v))
                            .collect();
                        format!("[{}...]", dims.join(", "))
                    }
                    2 => format!("{:.4}", self.datetime_values[idx]),
                    3 => format!("{:.4}", self.boolean_values[idx]),
                    _ => "?".to_string(),
                };

                let flags = format!(
                    "{}{}",
                    if self.masks[idx] { "MASKED " } else { "" },
                    if self.is_task_node[idx] { "TASK" } else { "" }
                );

                println!("  [{:4}] {:10} = {:30} {}", i, sem_type, value_str, flags);

                printed += 1;
                if printed >= 50 {
                    println!("  ... and {} more cells", seq_non_padding - printed);
                    break;
                }
            }

            // Show attention mask summary for first sequence
            if seq_idx == 0 {
                println!();
                println!("  Attention Mask Summary (first 20x20):");
                let mask_start = seq_idx * seq_len * seq_len;

                // Column attention
                print!("    Column attn (1s): ");
                let col_ones: usize = self.column_attn_mask
                    [mask_start..mask_start + seq_len * seq_len]
                    .iter()
                    .filter(|&&v| v)
                    .count();
                println!(
                    "{} / {} = {:.2}%",
                    col_ones,
                    seq_len * seq_len,
                    100.0 * col_ones as f64 / (seq_len * seq_len) as f64
                );

                // Feature attention
                print!("    Feature attn (1s): ");
                let feat_ones: usize = self.feature_attn_mask
                    [mask_start..mask_start + seq_len * seq_len]
                    .iter()
                    .filter(|&&v| v)
                    .count();
                println!(
                    "{} / {} = {:.2}%",
                    feat_ones,
                    seq_len * seq_len,
                    100.0 * feat_ones as f64 / (seq_len * seq_len) as f64
                );

                // Neighbor attention
                print!("    Neighbor attn (1s): ");
                let nbr_ones: usize = self.neighbor_attn_mask
                    [mask_start..mask_start + seq_len * seq_len]
                    .iter()
                    .filter(|&&v| v)
                    .count();
                println!(
                    "{} / {} = {:.2}%",
                    nbr_ones,
                    seq_len * seq_len,
                    100.0 * nbr_ones as f64 / (seq_len * seq_len) as f64
                );

                // Print first 128x128 of each attention mask
                let mask_size = 128.min(seq_len);

                println!();
                println!(
                    "    Column attention mask ({}x{}, . = 0, # = 1):",
                    mask_size, mask_size
                );
                for q in 0..mask_size {
                    print!("      ");
                    for kv in 0..mask_size {
                        let midx = mask_start + q * seq_len + kv;
                        print!(
                            "{}",
                            if self.column_attn_mask[midx] {
                                '#'
                            } else {
                                '.'
                            }
                        );
                    }
                    println!();
                }

                println!();
                println!(
                    "    Feature attention mask ({}x{}, . = 0, # = 1):",
                    mask_size, mask_size
                );
                for q in 0..mask_size {
                    print!("      ");
                    for kv in 0..mask_size {
                        let midx = mask_start + q * seq_len + kv;
                        print!(
                            "{}",
                            if self.feature_attn_mask[midx] {
                                '#'
                            } else {
                                '.'
                            }
                        );
                    }
                    println!();
                }

                println!();
                println!(
                    "    Neighbor attention mask ({}x{}, . = 0, # = 1):",
                    mask_size, mask_size
                );
                for q in 0..mask_size {
                    print!("      ");
                    for kv in 0..mask_size {
                        let midx = mask_start + q * seq_len + kv;
                        print!(
                            "{}",
                            if self.neighbor_attn_mask[midx] {
                                '#'
                            } else {
                                '.'
                            }
                        );
                    }
                    println!();
                }
            }
            println!();
        }

        println!(
            "╔══════════════════════════════════════════════════════════════════════════════╗"
        );
        println!(
            "║                           END OF BATCH DEBUG DUMP                            ║"
        );
        println!(
            "╚══════════════════════════════════════════════════════════════════════════════╝"
        );
    }

    fn into_pyobject(self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        Ok(vec![
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
            (
                "semantic_types",
                PyArray1::from_vec(py, self.semantic_types),
            )
                .into_py_any(py)?,
            ("masks", PyArray1::from_vec(py, self.masks)).into_py_any(py)?,
            ("is_task_node", PyArray1::from_vec(py, self.is_task_node)).into_py_any(py)?,
            ("is_padding", PyArray1::from_vec(py, self.is_padding)).into_py_any(py)?,
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
            ("true_batch_size", self.true_batch_size).into_py_any(py)?,
        ])
    }
}

/// A seed row for BFS sampling.
struct SamplerItem {
    database_idx: usize,
    row_idx: RowIdx,
    target_column: ColumnIdx,
    columns_to_drop: Vec<ColumnIdx>,
    is_task_table: bool,
}

#[pyclass]
pub struct Sampler {
    databases: Vec<Database>,
    items: Vec<SamplerItem>,
    batch_size: usize,
    seq_len: usize,
    rank: usize,
    world_size: usize,
    max_bfs_width: usize,
    d_text: usize,
    seed: u64,
    epoch: u64,
}

#[pymethods]
impl Sampler {
    /// Create a new Sampler for multi-database training.
    ///
    /// Args:
    ///     db_configs: List of (db_path, task_table_idx, target_column_idx, columns_to_drop)
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
            let database = Database::load(Path::new(&db_path)).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "Failed to load database '{}': {}",
                    db_path, e
                ))
            })?;

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

    fn len_py(&self) -> usize {
        self.items.len().div_ceil(self.batch_size * self.world_size)
    }

    fn batch_py(&self, py: Python<'_>, batch_idx: usize) -> PyResult<Vec<Py<PyAny>>> {
        self.batch(batch_idx).into_pyobject(py)
    }

    fn shuffle_py(&mut self, epoch: u64) {
        self.epoch = epoch;
        let mut rng = StdRng::seed_from_u64(epoch.wrapping_add(self.seed));
        self.items.shuffle(&mut rng);
    }
}

impl Sampler {
    fn batch(&self, batch_idx: usize) -> BatchVecs {
        let start_idx = batch_idx * self.batch_size * self.world_size + self.rank * self.batch_size;
        let true_batch_size = self
            .batch_size
            .min(self.items.len().saturating_sub(start_idx));

        let mut vecs = unsafe { BatchVecs::new(self.batch_size, self.seq_len, self.d_text) };
        vecs.true_batch_size = true_batch_size;
        vecs.is_padding.fill(true);

        self.fill_batch_vecs(&mut vecs, start_idx);
        vecs
    }

    fn fill_batch_vecs(&self, vecs: &mut BatchVecs, start_idx: usize) {
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

        (0..self.batch_size).into_par_iter().for_each(|i| {
            let j = (start_idx + i) % items_len;
            let item = &self.items[j];
            let db = &self.databases[item.database_idx];

            let seq_offset = i * seq_len;
            let text_offset = i * seq_len * d_text;
            let mask_offset = i * seq_len * seq_len;

            // Each thread gets its own index buffers for attention mask computation
            let mut indices = SequenceIndices::new(seq_len);

            // Get thread-local traversal buffers (reused across sequences)
            let mut trav = get_traversal_buffers(db.num_rows());

            unsafe {
                let slice = SequenceSlice {
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

                self.fill_sequence(item, slice, &mut indices, &mut trav);
            }

            // Return buffers to thread-local pool for reuse
            return_traversal_buffers(trav);
        });
    }

    fn fill_sequence(
        &self,
        item: &SamplerItem,
        mut seq: SequenceSlice<'_>,
        idx: &mut SequenceIndices,
        trav: &mut TraversalBuffers,
    ) {
        let db = &self.databases[item.database_idx];
        let seed_row = db.get_row(item.row_idx);
        let seed_table = db.get_table(seed_row.table_idx);
        let seed_timestamp = self.get_row_timestamp(db, item.row_idx);

        // Use pre-allocated buffers from trav (already reset)
        trav.f2p_frontier.push((0, item.row_idx));

        let mut seq_i = 0;
        let mut rng = StdRng::seed_from_u64(
            self.epoch
                .wrapping_add(item.row_idx.0 as u64)
                .wrapping_add(self.seed),
        );

        loop {
            // Select next node: prioritize f2p (parent) edges
            let (depth, row_idx) = if let Some(frontier_item) = trav.f2p_frontier.pop() {
                frontier_item
            } else {
                let mut found = None;
                for (d, nodes) in trav.p2f_frontier.iter().enumerate() {
                    if !nodes.is_empty() {
                        found = Some(d);
                        break;
                    }
                }
                if let Some(d) = found {
                    let r = rng.random_range(0..trav.p2f_frontier[d].len());
                    let len = trav.p2f_frontier[d].len();
                    trav.p2f_frontier[d].swap(r, len - 1);
                    (d, trav.p2f_frontier[d].pop().unwrap())
                } else {
                    break;
                }
            };

            if trav.visited.contains(row_idx.0 as usize) {
                continue;
            }
            trav.visited.insert(row_idx.0 as usize);

            let row = db.get_row(row_idx);
            let table = db.get_table(row.table_idx);

            // Collect f2p neighbors (rows this row points TO via FK)
            trav.f2p_neighbors.clear();
            for &edge_idx in &db.edges_from[row_idx.0 as usize] {
                let edge = &db.fk_edges[edge_idx];
                trav.f2p_neighbors.push(edge.to_row);
                trav.f2p_frontier.push((depth + 1, edge.to_row));
            }

            // Follow reverse edges (p2f: child rows -> this row)
            trav.children.clear();
            for &edge_idx in &db.edges_to[row_idx.0 as usize] {
                let edge = &db.fk_edges[edge_idx];

                // Temporal constraint: don't include future rows
                if let Some(cutoff) = seed_timestamp
                    && !db.row_is_before(edge.from_row, cutoff)
                {
                    continue;
                }

                let child_table = db.get_table(db.get_row(edge.from_row).table_idx);
                if child_table.idx == seed_table.idx || item.is_task_table {
                    trav.children.push(edge.from_row);
                }
            }

            // Subsample if too many children
            let num_children = trav.children.len();
            let sample_count = num_children.min(self.max_bfs_width);

            if num_children > self.max_bfs_width {
                // Shuffle first sample_count elements using Fisher-Yates partial shuffle
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
            let col_start = table.column_range.0.0;
            let is_seed_row = row_idx == item.row_idx;

            for (local_idx, normalized_cell) in row.normalized.iter().enumerate() {
                let global_col_idx = ColumnIdx(col_start + local_idx as u32);
                let column = db.get_column(global_col_idx);

                if is_seed_row && item.columns_to_drop.contains(&global_col_idx) {
                    continue;
                }
                if matches!(normalized_cell, NormalizedCellValue::Null) {
                    continue;
                }

                // Fill indices (for attention mask computation)
                idx.node[seq_i] = row_idx.0 as i32;
                idx.table[seq_i] = row.table_idx.0 as i32;
                idx.column[seq_i] = global_col_idx.0 as i32;

                for (j, &neighbor_row) in trav
                    .f2p_neighbors
                    .iter()
                    .take(MAX_F2P_NEIGHBORS)
                    .enumerate()
                {
                    idx.f2p_neighbors[seq_i * MAX_F2P_NEIGHBORS + j] = neighbor_row.0 as i32;
                }

                // Fill values
                seq.semantic_types[seq_i] = column.dtype as i32;

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
                        if let Some(embedding) = db.text_value_embeddings.get(text_idx.0 as usize) {
                            let start = seq_i * self.d_text;
                            let end = start + self.d_text.min(embedding.len());
                            seq.text_values[start..end].copy_from_slice(&embedding[..end - start]);
                        }
                    }
                    _ => {}
                }

                if let Some(ref embedding) = column.column_description_embedding {
                    let start = seq_i * self.d_text;
                    let end = start + self.d_text.min(embedding.len());
                    seq.column_name_values[start..end].copy_from_slice(&embedding[..end - start]);
                }

                seq.masks[seq_i] = is_seed_row && global_col_idx == item.target_column;
                seq.is_task_node[seq_i] =
                    item.is_task_table && (row.table_idx == db.get_row(item.row_idx).table_idx);
                seq.is_padding[seq_i] = false;

                seq_i += 1;
                if seq_i >= self.seq_len {
                    break;
                }
            }

            if seq_i >= self.seq_len {
                break;
            }
        }

        self.compute_attention_masks(&mut seq, idx);
    }

    #[cfg(target_arch = "x86_64")]
    fn compute_attention_masks(&self, seq: &mut SequenceSlice<'_>, idx: &SequenceIndices) {
        use std::arch::x86_64::*;

        let seq_len = self.seq_len;

        unsafe {
            let minus_one = _mm256_set1_epi32(-1);

            for q in 0..seq_len {
                if seq.is_padding[q] {
                    continue;
                }

                let q_node = idx.node[q];
                let q_table = idx.table[q];
                let q_col = idx.column[q];

                let q_node_v = _mm256_set1_epi32(q_node);
                let q_table_v = _mm256_set1_epi32(q_table);
                let q_col_v = _mm256_set1_epi32(q_col);

                let q_f2p_start = q * MAX_F2P_NEIGHBORS;
                let q_f2p: [__m256i; MAX_F2P_NEIGHBORS] =
                    std::array::from_fn(|i| _mm256_set1_epi32(idx.f2p_neighbors[q_f2p_start + i]));

                let mask_row_start = q * seq_len;
                let mut kv = 0;
                let simd_len = seq_len / 8 * 8;

                while kv < simd_len {
                    let kv_nodes = _mm256_loadu_si256(idx.node.as_ptr().add(kv) as *const __m256i);
                    let kv_tables =
                        _mm256_loadu_si256(idx.table.as_ptr().add(kv) as *const __m256i);
                    let kv_cols = _mm256_loadu_si256(idx.column.as_ptr().add(kv) as *const __m256i);

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
                        let kv_neighbors =
                            _mm256_i32gather_epi32::<4>(idx.f2p_neighbors.as_ptr(), offset_indices);
                        let matches = _mm256_cmpeq_epi32(kv_neighbors, q_node_v);
                        nbr_result = _mm256_or_si256(nbr_result, matches);
                    }

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

                // Scalar fallback for remainder
                while kv < seq_len {
                    if !seq.is_padding[kv] {
                        let kv_node = idx.node[kv];
                        let kv_table = idx.table[kv];
                        let kv_col = idx.column[kv];
                        let mask_idx = mask_row_start + kv;

                        seq.column_attn_mask[mask_idx] = (q_col == kv_col) && (q_table == kv_table);

                        let same_node = q_node == kv_node;
                        let kv_in_q_f2p = (0..MAX_F2P_NEIGHBORS).any(|i| {
                            let n = idx.f2p_neighbors[q_f2p_start + i];
                            n >= 0 && n == kv_node
                        });
                        seq.feature_attn_mask[mask_idx] = same_node || kv_in_q_f2p;

                        let kv_f2p_start = kv * MAX_F2P_NEIGHBORS;
                        let q_in_kv_f2p = (0..MAX_F2P_NEIGHBORS)
                            .any(|i| idx.f2p_neighbors[kv_f2p_start + i] == q_node);
                        seq.neighbor_attn_mask[mask_idx] = q_in_kv_f2p;
                    }
                    kv += 1;
                }
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn compute_attention_masks(&self, seq: &mut SequenceSlice<'_>, idx: &SequenceIndices) {
        let seq_len = self.seq_len;

        for q in 0..seq_len {
            if seq.is_padding[q] {
                continue;
            }

            let q_node = idx.node[q];
            let q_table = idx.table[q];
            let q_col = idx.column[q];
            let q_f2p_start = q * MAX_F2P_NEIGHBORS;

            for kv in 0..seq_len {
                if seq.is_padding[kv] {
                    continue;
                }

                let kv_node = idx.node[kv];
                let kv_table = idx.table[kv];
                let kv_col = idx.column[kv];
                let mask_idx = q * seq_len + kv;

                seq.column_attn_mask[mask_idx] = (q_col == kv_col) && (q_table == kv_table);

                let same_node = q_node == kv_node;
                let kv_in_q_f2p = (0..MAX_F2P_NEIGHBORS).any(|i| {
                    let n = idx.f2p_neighbors[q_f2p_start + i];
                    n >= 0 && n == kv_node
                });
                seq.feature_attn_mask[mask_idx] = same_node || kv_in_q_f2p;

                let kv_f2p_start = kv * MAX_F2P_NEIGHBORS;
                let q_in_kv_f2p =
                    (0..MAX_F2P_NEIGHBORS).any(|i| idx.f2p_neighbors[kv_f2p_start + i] == q_node);
                seq.neighbor_attn_mask[mask_idx] = q_in_kv_f2p;
            }
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_batch_generation() {
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

        let batch = sampler.batch(0);

        // Verify dimensions
        let l = batch_size * seq_len;
        assert_eq!(batch.number_values.len(), l);
        assert_eq!(batch.text_values.len(), l * d_text);
        assert_eq!(batch.column_attn_mask.len(), batch_size * seq_len * seq_len);

        // Verify some content was filled
        let non_padding: usize = batch.is_padding.iter().filter(|&&p| !p).count();
        assert!(non_padding > 0, "Should have non-padding positions");

        let masked: usize = batch.masks.iter().filter(|&&m| m).count();
        assert!(masked > 0, "Should have masked (target) positions");
    }

    #[test]
    fn test_performance() {
        use std::time::Instant;

        let db_path = "sample_data_f1.rkyv";
        if !Path::new(db_path).exists() {
            eprintln!("Skipping: {} not found", db_path);
            return;
        }

        let batch_size = 32;
        let seq_len = 1024;
        let d_text = 768;
        let num_batches = 2000;

        let sampler = create_test_sampler(
            vec![(db_path.to_string(), 7, 51, vec![])],
            batch_size,
            seq_len,
            d_text,
        );

        // Warmup
        let _ = sampler.batch(0);

        // Timed run
        let start = Instant::now();
        for i in 0..num_batches {
            let _ = sampler.batch(i % sampler.items.len().div_ceil(batch_size));
        }
        let elapsed = start.elapsed();

        let total_sequences = num_batches * batch_size;
        let ms_per_batch = elapsed.as_secs_f64() * 1000.0 / num_batches as f64;
        let seqs_per_sec = total_sequences as f64 / elapsed.as_secs_f64();

        println!(
            "\n=== Performance ({} batches of {}) ===",
            num_batches, batch_size
        );
        println!("Total time:      {:>8.2} s", elapsed.as_secs_f64());
        println!("Per batch:       {:>8.2} ms", ms_per_batch);
        println!("Sequences/sec:   {:>8.0}", seqs_per_sec);
        println!("Rayon threads:   {:>8}", rayon::current_num_threads());
    }

    #[test]
    fn test_single_trajectory_debug() {
        let db_path = "sample_data_f1.rkyv";
        if !Path::new(db_path).exists() {
            eprintln!("Skipping: {} not found", db_path);
            return;
        }

        // Single trajectory: batch_size = 1
        let batch_size = 1;
        let seq_len = 128; // Smaller seq_len for readability
        let d_text = 768;

        let sampler = create_test_sampler(
            vec![(db_path.to_string(), 7, 51, vec![])],
            batch_size,
            seq_len,
            d_text,
        );

        // Generate one trajectory
        let batch = sampler.batch(0);

        // Debug dump
        batch.dump_debug(seq_len, d_text);

        // Also print the seed item info
        println!("\n=== Seed Item Info ===");
        let item = &sampler.items[0];
        let db = &sampler.databases[item.database_idx];
        let row = db.get_row(item.row_idx);
        let table = db.get_table(row.table_idx);
        let target_col = db.get_column(item.target_column);

        println!("Database idx: {}", item.database_idx);
        println!("Seed row idx: {}", item.row_idx.0);
        println!("Seed table: {} (idx {})", table.name, row.table_idx.0);
        println!(
            "Target column: {} (idx {})",
            target_col.name, item.target_column.0
        );
        println!("Columns to drop: {:?}", item.columns_to_drop);

        // Print raw values of seed row
        println!("\nSeed row raw values:");
        for (i, cell) in row.raw.iter().enumerate() {
            let col_idx = table.column_range.0.0 + i as u32;
            let col = db.get_column(ColumnIdx(col_idx));
            println!("  {}: {:?}", col.name, cell);
        }
    }
}
