# README

This rust crate is designed to do two things:

1) Process databases into our sampler-friendly graph format
2) Create pre-batched samples of data for our Relational Transformer pytorch model.


## Preprocessing

Each input database (for pre-training, or fine-tuning) is expected to be represented as a directory with the following (example) structure:

```
(database_name)
├── tables
│   ├── (table1).parquet
│   ├── (table2).parquet
...
├── metadata.json
```

There should be one parquet file for each table in the database, and a metadata.json file with the following (example) structure.

```
{
    "user": {
        "primary_key_column": "id",
        "foreign_key_column_to_primary_key_table": {},
        "time_column": null,
        "categorical_columns": [],
        "ignored_columns": [],
    },
    "movie": {
        "primary_key_column": "id",
        "foreign_key_column_to_primary_key_table": {},
        "time_column": null,
        "categorical_columns": [],
        "ignored_columns": [],
    },
    "rating": {
        "primary_key_column": "id",
        "foreign_key_column_to_primary_key_table": {
            "user_id": "user", 
            "movie_id": "movie"
        },
        "time_column": "rated_at",
        "categorical_columns": [],
        "ignored_columns": [],
    }
}
```

This metadata describes, on a per-table level, our special semantics.
When the database is preprocessed, we'll ignore all of the column names in the "ignored_columns" list for each table.
Additionally, each table gets one special "time_column" that indicates when the column's data became relevant.
Think "created_at" for facts.
We also mark out which columns should be treated as "categorical" by the preprocessor - the underlying datatype on these
might be text, integer, or boolean, but semantically they might represent a "categorical" variable that we want to
present to the foundation model.

The preprocessing script will write out a "(database_name).rkyv" file that captures the graph structure of the database,
along with all of the values (normalized) and an "interned" text embedding for each distinct string seen.

You should preprocess *all* databases you want to pre-train on, and store them in a directory of training databases:

Fine-tuning a model involves passing a *single* one of these .rkyv (preprocessed) files, representing the new database
to fine-tune on.

```
preprocessed_dbs
├── sample_f1.rkyv
├── movie_ratings.rkyv
...
```


## Sampling

We expose (through PyO3) a rust object that can be used to play the role of a dataloader in PyTorch. This sampler does two things:

1) Performs BFS-based neighborhood sampling around a target row in some database in the training set.
2) Pre-computes *attention mask* tensors on the CPU side.