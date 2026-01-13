//! Tributary: Fast preprocessing and sampling for relational databases.
//!
//! This crate provides:
//! - Types for representing databases as graphs
//! - Preprocessing from parquet files + metadata.json
//! - BFS-based neighborhood sampling for ML models

use mimalloc::MiMalloc;
use pyo3::prelude::*;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod embedder;
pub mod sampler;
pub mod types;
pub mod utility;

// Re-export common types at crate root
pub use types::{
    ArchivedColumnExt, ArchivedTableExt, Cells, Column, ColumnIdx, CsrGraph, Database,
    DatabaseMetadata, EmbeddingIdx, Graph, Manifest, ManifestStats, NO_TIMESTAMP, PACKED_NULL,
    PackedCell, PreprocessingContext, RowIdx, Schema, SemanticType, Table, TableIdx, TableMetadata,
    is_packed_null, load_metadata, pack_embedding_idx, pack_null, pack_numerical, pack_timestamp,
};
pub use utility::{TIMESTAMP_DIM, expand_timestamp};

pub use embedder::{EMBEDDING_DIM, Embedder, EmbedderConfig, EmbedderError};

pub use sampler::{BatchVecs, Sampler, SamplerConfig, Split};

#[pymodule]
fn tributary(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<sampler::Sampler>()?;
    Ok(())
}
