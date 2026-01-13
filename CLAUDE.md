# Start here if you're Claude!

This repo contains code designed to build a "relational transformer" machine learning model.

There are *two* main pieces:

## Python-side

The python code (pytorch, model definition, and training script) live in the `dbtransformer` directory.
This directory uses the manager `uv` for all python related things.

Important! Any time you want to use python to do some temporary scripting related things, you SHOULD USE `uv` to run commands.


## Rust-side

The rust code (data preprocessing and on-the-fly sampling) lives in the `tributary` directory.
This directory uses `cargo` like a normal rust project.

## Data

The datasets for this project currently live in a Google Cloud Storage bucket called "dbtransformer".
This bucket has two "subdirectories":

(1) gs://dbtransformer/databases_raw/

This "raw" directory contains a number of databases - one subdirectory per database.
For example, you can see the database "rel-event" here:

gs://dbtransformer/database_raw/rel-event/

rel-event/
  - metadata.json
  - db/
    - event_attendees.parquet
    - event_interest.parquet
    - events.parquet
    - user_friends.parquet
    - users.parquet

The structure of these individual, un-processed database directories includes a "metadata.json" file that defines some
extra structure for the database - foreign key relationships, semantic type overrides, and more. 
In the db directory, there's one .parquet file per table in the relational database.


(2)  gs://dbtransformer/databases_preprocessed/

We use a preprocessing script (see `tributary`'s `preprocessor.rs` for this) to convert these directories into
preprocessed files that can be used by the sampler.

### Preprocessed Output Format

Each preprocessed database consists of two files:

```
databases_preprocessed/
├── rel-event.rkyv           # Schema, graph (CSR), cells, timestamps (~2.5GB for rel-event)
└── rel-event.embeddings.bin # Text embeddings, mmap'd separately (~165MB for rel-event)
```

The split format is designed for memory efficiency:
- `.rkyv` contains schema, graph structure, and cell values (loads into RAM)
- `.embeddings.bin` contains text embeddings (mmap'd - OS only pages in accessed embeddings)

For large datasets like rel-amazon with 31M unique text values (~47GB of embeddings), this split approach
allows the sampler to work without loading all embeddings into RAM.

### Running the Preprocessor

```bash
cd tributary
cargo run --release --bin preprocessor -- \
  --input-dir ~/gcs/databases_raw/rel-event \
  --output-dir ~/gcs/databases_preprocessed/ \
  --verbose
```

The preprocessor streams through the data in chunks (configurable with `--chunk-size`) to handle
large datasets without running out of memory.

This bucket is mounted locally with GCSFUSE at ~/gcs so you can interact with the data there.