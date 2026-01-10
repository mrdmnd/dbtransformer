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

The preprocessing script will write out a "(database_name).rkyv" file that captures the graph structure of the database,
along with all of the values (normalized) and a text embedding for each distinct string value seen in the data.

You should preprocess *all* databases you want to pre-train (or evaluate) on, and place them into the "databases_preprocessed" directory.

Fine-tuning a model involves passing a *single* one of these .rkyv (preprocessed) files, representing the new database
to fine-tune on.

```bash
databases_preprocessed
├── sample_f1.rkyv
├── movie_ratings.rkyv
...
```

Example usage:

```bash
cargo run --release --bin preprocessor -- \
  --input-database-dir=databases_raw/rel-event \
  --output-dir=databases_preprocessed/ \
  --verbose
```

## Sampling

We expose (through PyO3) a rust object that can be used to play the role of a dataloader in PyTorch. This sampler does two things:

1) Performs BFS-based neighborhood sampling around a target row in some database in the training set.
2) Pre-computes *attention mask* tensors on the CPU side.
