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





Maintenance bits for me to remember (others should ignore):

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
tributary preprocessing: take some directory which looks like

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
```

then run

mrdmnd@khadgar:~/dbtransformer/tributary$ cargo run --release --bin preprocessor -- --data-dir=sample_data_f1 --verbose

Then, you can sample from it



## dataset gathering

```
import pooch
from loguru import logger
from relbench.datasets import get_dataset, get_dataset_names
from relbench.tasks import get_task, get_task_names
from tqdm import tqdm

if __name__ == "__main__":
    logger.info("Downloading all RelBench datasets and tasks")

    cache_dir = f"{pooch.os_cache('relbench')}"
    logger.info(f"Cache: {cache_dir}")

    for dataset_name in tqdm(get_dataset_names(), colour="green"):
        logger.info(f"Downloading dataset: {dataset_name}")
        get_dataset(dataset_name, download=True)
```
