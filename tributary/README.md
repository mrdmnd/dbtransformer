# README

This rust crate is designed to do two things:

1) Process databases into our sampler-friendly graph format
2) Create pre-batched samples of data for our Relational Transformer pytorch model.

## Preprocessing

Each input database (for pre-training, or fine-tuning) is expected to be represented as a directory with the following (example) structure.
These live in cloud storage, in a bucket: `gs://dbtransformer/databases_raw`

```bash
(database_name)
├── db
│   ├── (table1).parquet
│   ├── (table2).parquet
...
├── metadata.json
```

There should be one parquet file for each table in the database, and a metadata.json file with the following (example) structure.

```javascript
{
    "user": {
        "primary_key_column": "id",
        "foreign_key_column_to_primary_key_table": {},
        "time_column": null,
        "stype_overrides": {"status": "categorical"},
        "ignored_columns": [],
    },
    "movie": {
        "primary_key_column": "id",
        "foreign_key_column_to_primary_key_table": {},
        "time_column": null,
        "stype_overrides": {},
        "ignored_columns": [],
    },
    "rating": {
        "primary_key_column": null,
        "foreign_key_column_to_primary_key_table": {
            "user_id": "user",
            "movie_id": "movie"
        },
        "time_column": "rated_at",
        "stype_overrides": {},
        "ignored_columns": [],
    }
}
```

This metadata describes, on a per-table level, our database semantics.

When the database is preprocessed, we'll ignore all of the column names in the "ignored_columns" list for each table.

Additionally, each table is allowed to have up to one special "time_column" that indicates when the column's data became relevant -- think "created_at" for facts.

Columns are automatically assigned a semantic type based on their underlying datatype, but in some cases, we may want to override this. This is done via the "stype_overrides" field in the metadata.json file.

Most commonly, this would be for columns whose Polars dtype is string, but semantically represent a categorical variable.

Another circumstance where we might want to do this override is for columns whose Polars dtype is UInt, but semantically are an identifier, and should be treated as such (instead of a numerical variable, or categorical variable).

## Output Format

The preprocessing script outputs a directory per database:

```bash
databases_preprocessed/
├── rel-event/
│   ├── manifest.json       # Metadata, stats, file checksums
│   ├── schema.rkyv         # Tables, columns, column embeddings
│   ├── graph.rkyv          # CSR adjacency (outgoing + incoming edges)
│   ├── cells.rkyv          # Cell values, row offsets, row timestamps
│   └── embeddings.bin      # Text embeddings (f16), mmap'd separately
├── movie_ratings/
│   ├── manifest.json
│   ├── schema.rkyv
│   ├── graph.rkyv
│   ├── cells.rkyv
│   └── embeddings.bin
...
```

**Why split files?**

- **schema.rkyv** - Small (KB). Tables, columns, normalization stats. Always loaded.
- **graph.rkyv** - Medium. CSR adjacency for FK edges. Memory-mapped.
- **cells.rkyv** - Large. Packed cell values, row timestamps. Memory-mapped.
- **embeddings.bin** - Huge. Text embeddings for all unique text/categorical values. Memory-mapped so the OS only pages in embeddings that are actually accessed during sampling.

For large datasets like rel-amazon with 31M unique texts (~47GB of embeddings), this split approach is essential for memory efficiency. During sampling, only ~0.2% of embeddings are accessed per batch.

**Streaming Architecture:**

The preprocessor uses a streaming architecture to handle large datasets:

1. **Schema Discovery** - Scan parquet metadata for column types and statistics
2. **Vocabulary Extraction** - Stream through text columns, deduplicate, write to disk
3. **Embedding** - Stream vocab file through embedder, write directly to `embeddings.bin`
4. **Cell Encoding** - Stream tables, encode cells with vocab lookup
5. **FK Edge Building** - Stream FK columns, resolve to row indices, build CSR graph

Memory usage is O(chunk_size) regardless of dataset size.

**Example usage:**

```bash
cargo run --release --bin preprocessor -- \
  --input-dir databases_raw/rel-event \
  --output-dir databases_preprocessed/ \
  --chunk-size 100000 \
  --verbose
```

**Loading in Rust:**

```rust
use tributary::Database;

// Loads all split files, mmaps each component
let db = Database::load("databases_preprocessed/rel-event")?;

// Access schema (in memory)
let table = db.table(TableIdx(0));

// Access graph (mmap'd)
let neighbors = db.outgoing_neighbors(row_idx);

// Access cells (mmap'd)
let cells = db.row_cells(row_idx);

// Access embeddings (mmap - only accessed pages loaded)
let embedding = db.get_embedding(embedding_idx);
```

## Sampling

We expose (through PyO3) a rust object that can be used to play the role of a dataloader in PyTorch. This sampler does two things:

1) Performs BFS-based neighborhood sampling around a target row in some database in the training set.
2) Pre-computes *attention mask* tensors on the CPU side.

**Python usage:**

```python
import tributary

# Load from a database directory
sampler = tributary.Sampler(
    db_path="databases_preprocessed/rel-event",
    batch_size=32,
    seq_len=1024,
    max_bfs_width=256,
    seed=42,
    num_threads=4,  # Set to num_cpus / world_size for multi-process training
)

# Iterate over batches
for batch_idx in range(sampler.len_py()):
    batch = sampler.batch_py(batch_idx)
    # ... training loop
```
