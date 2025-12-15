# dbtransformer

This is a HACKATHON project - expect rough edges and incomplete things.

Goal: train a relational transformer model!

Note: this is ONLY designed to run on CUDA-enabled machines.
Currently presupposes a pytorch 2.9 environment with cuda 13.0, to better support Blackwell cards.

It's pretty important to run on a good CPU setup too, because one of the major bottlenecks is loading batches into the
GPU at runtime, which is done by a custom Rust implementation (highly optimized, multithreaded BFS + low allocations)

## Datasets

- [RelBench](https://huggingface.co/datasets/relbench/relbench)

## Setup

```bash
pre-commit clean
pre-commit install
pre-commit install-hooks
uv sync --extra flash
```

(do the --extra flash on cuda machines, just --sync otherwise, to get flash-attention appropriately)

## Run Locally

```bash
uv run torchrun --nproc_per_node=1 dbtransformer/bin/train.py
```

## Profiling

You can profile with

```bash
uv run torchrun --nproc_per_node=1 dbtransformer/bin/train.py --profile torch --no-wandb
```

Then open `http://ui.perfetto.dev` in your browser and load the content from the `profiler_logs` directory.

In another process you can wandb login and then
`wandb beta leet` if you want a sick TUI, or go to the website version at:

<https://wandb.ai/mttrdmnd-massachusetts-institute-of-technology/dbtransformer?nw=nwusermttrdmnd>

Maintenance bits for me to remember:

## Remote

### Build Docker Image

(only needs to be done once, I already did this)
The image also has a startup script which pulls the latest repo down and then uv syncs
the dependencies.

docker login
docker build --platform linux/amd64 -t mredmondhex/dbtransformer:latest .
docker push mredmondhex/dbtransformer:latest

### Start the Instance

Do it through the prime intellect thing

### Connect to the Instance

ssh -i ~/.ssh/primeintellect_ed25519 -p 42069 root@<ip>
cd /
uv run torchrun --nproc_per_node=1 dbtransformer/bin/train.py

### get profiler traces to local storage

rm -rf profiler_log_dump && scp -i ~/.ssh/primeintellect_ed25519 -P 42069 -r root@62.169.159.172:/app/profiler_logs ./profiler_log_dump



## Rust Stuff
batcher preprocessing: take some directory which looks like

```
sample_data_f1
├── db
│   ├── circuits.parquet
│   ├── constructor_results.parquet
│   ├── constructor_standings.parquet
│   ├── constructors.parquet
│   ├── drivers.parquet
│   ├── qualifying.parquet
│   ├── races.parquet
│   ├── results.parquet
│   └── standings.parquet
└── tasks
    ├── driver-dnf
    │   ├── test.parquet
    │   ├── train.parquet
    │   └── val.parquet
    ├── driver-position
    │   ├── test.parquet
    │   ├── train.parquet
    │   └── val.parquet
    └── driver-top3
        ├── test.parquet
        ├── train.parquet
        └── val.parquet
```

then run

mrdmnd@khadgar:~/dbtransformer/batcher$ cargo run --release --bin preprocessor -- --data-dir=sample_data_f1 --verbose

which will create a sample_data_f1.rkyv file (for the database, plus the task tables) that is basically the graph representation as a zero copy rust struct.

Then, you can sample from it

example:

mrdmnd@khadgar:~/dbtransformer/batcher$ cargo test test_performance --release -- --nocapture 2>&1
