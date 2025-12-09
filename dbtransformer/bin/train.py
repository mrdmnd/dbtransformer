"""
PyTorch Distributed Training Script
====================================
Always run with uv run torchrun:
    Single process:   uv run torchrun --nproc_per_node=1 train.py
    2 GPUs:           uv run torchrun --nproc_per_node=2 train.py
    8 GPUs:           uv run torchrun --nproc_per_node=8 train.py
    Multi-node:       uv run torchrun --nnodes=2 --node_rank=0 \
                      --master_addr=<IP> --master_port=29500 \
                      --nproc_per_node=2 train.py
"""

import os
import random

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grads_with_norm_, get_total_norm
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from tqdm import tqdm

import wandb

# Training config (will move to config file later)
MAX_GRAD_NORM = 1.0
SEED = 42
LR = 0.01
WEIGHT_DECAY = 1e-4
EPOCHS = 20
BATCH_SIZE = 256
HIDDEN_DIM = 2048
USE_COMPILE = True  # torch.compile for CUDA/MPS (PyTorch 2.0+)
USE_BFLOAT16 = False  # Set True for GPUs with bfloat16 support

# Weights & Biases config
USE_WANDB = True
WANDB_ENTITY: str = "mttrdmnd-massachusetts-institute-of-technology"
WANDB_PROJECT: str | None = "dbtransformer"  # Set to None to auto-generate


def seed_everything(seed: int = 42) -> None:
    """Set seeds for reproducibility across all random sources."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # TODO(mrdmnd): fix this lint
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup() -> tuple[int, int, torch.device]:
    """
    Initialize distributed training. Expects torchrun environment variables.

    Returns:
        (rank, world_size, device)
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    has_cuda = torch.cuda.is_available()

    backend = "nccl" if has_cuda else "gloo"
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Device selection: CUDA > MPS (single-process only) > CPU
    if has_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available() and world_size == 1:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"[Rank {rank}/{world_size}] backend={backend}, device={device}")
    return rank, world_size, device


def cleanup() -> None:
    """Clean up distributed process group."""
    dist.destroy_process_group()


def build_model(hidden_dim: int, device: torch.device) -> nn.Module:
    """Build a ~20M parameter MLP model."""
    model = nn.Sequential(
        nn.Linear(10, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    ).to(device)

    if USE_BFLOAT16 and device.type == "cuda":
        model = model.to(torch.bfloat16)

    return model


def configure_dynamo(use_compile: bool, world_size: int) -> None:
    """Configure torch.compile / dynamo settings."""
    if not use_compile:
        return
    # Increase cache size for complex models
    torch._dynamo.config.cache_size_limit = 64
    # Enable compiled autograd for DDP
    if world_size > 1:
        torch._dynamo.config.optimize_ddp = True


def wrap_with_ddp(
    model: nn.Module,
    world_size: int,
    device: torch.device,
) -> nn.Module:
    """Wrap model with DDP if using multiple processes."""
    if world_size <= 1:
        return model

    if device.type == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])
        return DistributedDataParallel(model, device_ids=[local_rank])
    return DistributedDataParallel(model)


def maybe_compile(
    model: nn.Module,
    use_compile: bool,
    device: torch.device,
) -> nn.Module:
    """Compile model with torch.compile if enabled and supported."""
    if use_compile and device.type in {"cuda", "mps"}:
        logger.info("Compiling model with torch.compile(dynamic=False)")
        return torch.compile(model, dynamic=False)  # type: ignore[return-value]
    return model


def create_dataloader(
    world_size: int,
    rank: int,
    batch_size: int,
    device: torch.device,
) -> tuple[DataLoader[tuple[torch.Tensor, ...]], DistributedSampler[TensorDataset]]:
    """Create synthetic dataset and distributed dataloader."""
    x = torch.randn(10000, 10)
    y = torch.randn(10000, 1)
    dataset = TensorDataset(x, y)

    sampler: DistributedSampler[TensorDataset] = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=(device.type == "cuda"),
    )
    return loader, sampler


def create_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    device: torch.device,
    total_steps: int,
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    """Create AdamW optimizer with OneCycleLR scheduler."""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        fused=(device.type == "cuda"),
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.2,
        anneal_strategy="linear",
    )
    return optimizer, scheduler


# TODO(mrdmnd): have fewer arguments in this function


# ruff: noqa: PLR0913, PLR0917
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
    device: torch.device,
    world_size: int,
) -> float:
    """Run one training epoch and return total loss."""
    total_loss = 0.0

    for batch_x, batch_y in loader:
        x = batch_x.to(device, non_blocking=True)
        y = batch_y.to(device, non_blocking=True)

        if USE_BFLOAT16 and device.type == "cuda":
            x = x.to(torch.bfloat16)
            y = y.to(torch.bfloat16)

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()

        # Gradient clipping (unwrap DDP if needed)
        if world_size > 1:
            params = list(model.module.parameters())  # type: ignore
        else:
            params = list(model.parameters())  # type: ignore
        grads = [p.grad for p in params if p.grad is not None]
        grad_norm = get_total_norm(grads)
        clip_grads_with_norm_(params, MAX_GRAD_NORM, grad_norm)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss


def sync_loss(
    total_loss: float,
    world_size: int,
    device: torch.device,
) -> float:
    """Synchronize loss across all ranks via all_reduce."""
    if world_size <= 1:
        return total_loss

    loss_tensor = torch.tensor([total_loss], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    return loss_tensor.item()


def init_wandb(rank: int) -> wandb.sdk.wandb_run.Run | None:
    """Initialize Weights & Biases on rank 0 only."""
    if not USE_WANDB or rank != 0:
        return None

    config = {
        "max_grad_norm": MAX_GRAD_NORM,
        "seed": SEED,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "use_compile": USE_COMPILE,
        "use_bfloat16": USE_BFLOAT16,
    }
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, config=config)
    logger.info(f"W&B run: {run.name}")
    return run


def main() -> None:
    seed_everything(SEED)
    rank, world_size, device = setup()

    # Initialize wandb on rank 0
    run = init_wandb(rank)

    # Configure dynamo before model creation
    configure_dynamo(USE_COMPILE, world_size)

    # Build and prepare model
    model = build_model(HIDDEN_DIM, device)
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {num_params:,} params (~{num_params / 1e6:.1f}M)")

    model = wrap_with_ddp(model, world_size, device)
    model = maybe_compile(model, USE_COMPILE, device)

    # Create data pipeline
    loader, sampler = create_dataloader(world_size, rank, BATCH_SIZE, device)

    # Create optimizer and scheduler
    total_steps = EPOCHS * len(loader)
    optimizer, scheduler = create_optimizer(model, LR, WEIGHT_DECAY, device, total_steps)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in tqdm(range(EPOCHS), disable=rank != 0):
        sampler.set_epoch(epoch)
        total_loss = train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device, world_size)
        total_loss = sync_loss(total_loss, world_size, device)

        if rank == 0:
            avg_loss = total_loss / len(loader)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, LR: {lr_now:.6f}")

            # Log to wandb
            if run is not None:
                wandb.log(
                    {
                        "loss": avg_loss,
                        "lr": lr_now,
                        "epoch": epoch + 1,
                    },
                    step=epoch + 1,
                )

    # Finish wandb run
    if run is not None:
        wandb.finish()

    cleanup()
    if rank == 0:
        print("Training complete!")


if __name__ == "__main__":
    main()
