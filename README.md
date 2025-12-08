# dbtransformer

Train a relational transformer on any database!

This is intended to be run both locally (for testing) and on Prime Intellect GPU servers.

## Datasets

- [RelBench](https://huggingface.co/datasets/relbench/relbench)
- [CTU Prague]
- ???

## Setup

```bash
pre-commit clean
pre-commit install
pre-commit install-hooks
uv sync
```

## Run Locally

```bash
uv run torchrun --nproc_per_node=1 dbtransformer/bin/train.py
```

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
