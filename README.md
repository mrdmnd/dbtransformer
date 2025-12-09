# dbtransformer

This is a HACKATHON project - expect rough edges and incomplete things.

Goal: train a relational transformer on any database!

## Datasets

- [RelBench](https://huggingface.co/datasets/relbench/relbench)

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
