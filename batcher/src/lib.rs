use mimalloc::MiMalloc;
use pyo3::prelude::*;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod embedder;
pub mod sampler;
pub mod types;

// Re-export commonly used types
pub use types::{
    Column, ColumnIdx, Database, ForeignKeyEdge, NormalizedCellValue, RawCellValue, Row, RowIdx,
    SemanticType, Table, TableIdx, TableType, TextIdx,
};

// Re-export embedder types
pub use embedder::{Embedder, EmbedderConfig, EmbedderError};

#[pymodule]
fn batcher(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<sampler::Sampler>()?;
    Ok(())
}
