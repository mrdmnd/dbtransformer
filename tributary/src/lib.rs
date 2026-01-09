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

// Re-export common types at crate root
pub use types::{
    CellValue, Column, ColumnIdx, Database, DatabaseMetadata, EmbeddingIdx, ForeignKeyEdge,
    Row, RowIdx, SemanticType, Table, TableIdx, TableMetadata, TIMESTAMP_DIM,
    load_metadata,
};

pub use embedder::{Embedder, EmbedderConfig, EmbedderError, EMBEDDING_DIM};

#[pymodule]
fn tributary(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<sampler::Sampler>()?;
    Ok(())
}
