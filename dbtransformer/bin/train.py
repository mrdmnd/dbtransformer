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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass
class TrainConfig:
    """Configuration for training."""

    max_grad_norm: float = 1.0
    seed: int = 42
    lr: float = 0.01
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 256
    hidden_dim: int = 2048
    use_compile: bool = True
    save_every: int = 5
    snapshot_path: str = "snapshot.pt"

    # Weights & Biases config
    use_wandb: bool = True
    wandb_entity: str = "mttrdmnd-massachusetts-institute-of-technology"
    wandb_project: str | None = "dbtransformer"


def seed_everything(seed: int = 42) -> None:
    """Set seeds for reproducibility across all random sources."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddp_setup() -> tuple[int, int, int, torch.device]:
    """
    Initialize distributed training.

    Returns:
        (local_rank, global_rank, world_size, device)
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ.get("RANK", "0"))

    has_cuda = torch.cuda.is_available()
    backend = "nccl" if has_cuda else "gloo"
    dist.init_process_group(backend=backend)

    world_size = dist.get_world_size()

    # Device selection: CUDA > MPS (single-process only) > CPU
    if has_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available() and world_size == 1:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"[Local Rank {local_rank} | Global Rank {global_rank}/{world_size}] backend={backend}, device={device}")
    return local_rank, global_rank, world_size, device


def build_model(hidden_dim: int, device: torch.device) -> nn.Module:
    """Build a ~20M parameter MLP model."""
    return nn.Sequential(
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


def configure_dynamo(use_compile: bool, world_size: int) -> None:
    """Configure torch.compile / dynamo settings."""
    if not use_compile:
        return
    torch._dynamo.config.cache_size_limit = 64
    if world_size > 1:
        torch._dynamo.config.optimize_ddp = True


def prepare_dataloader(
    dataset: TensorDataset,
    batch_size: int,
    world_size: int,
    rank: int,
    device: torch.device,
) -> tuple[DataLoader, DistributedSampler]:
    """Create distributed dataloader for the given dataset."""
    sampler: DistributedSampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
    )
    return loader, sampler


class Trainer:
    """Handles distributed training with checkpointing support."""

    # ruff: noqa: PLR0913, PLR0917
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        sampler: DistributedSampler,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        loss_fn: nn.Module,
        config: TrainConfig,
        device: torch.device,
        local_rank: int,
        global_rank: int,
        world_size: int,
    ) -> None:
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.device = device
        self.config = config

        self.train_loader = train_loader
        self.sampler = sampler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn

        self.epochs_run = 0
        self.wandb_run: wandb.sdk.wandb_run.Run | None = None

        # Wrap model with DDP
        self.model = self._wrap_model(model)

        # Optionally compile model
        if config.use_compile and device.type in {"cuda", "mps"}:
            logger.info("Compiling model with torch.compile(dynamic=False)")
            self.model = torch.compile(  # type: ignore[assignment]
                self.model, dynamic=False
            )

        # Load snapshot after DDP/compile so state dict keys match
        if Path(config.snapshot_path).exists():
            self._load_snapshot(config.snapshot_path)

    def _wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with DDP if using multiple processes."""
        if self.world_size <= 1:
            return model

        if self.device.type == "cuda":
            return DistributedDataParallel(model, device_ids=[self.local_rank])
        return DistributedDataParallel(model)

    def _load_snapshot(self, snapshot_path: str) -> None:
        """Load training state from a snapshot."""
        logger.info(f"Loading snapshot from {snapshot_path}")
        loc = f"cuda:{self.local_rank}" if self.device.type == "cuda" else "cpu"
        snapshot = torch.load(snapshot_path, map_location=loc)

        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]

        # Check if epochs were extended beyond original training
        saved_epochs = snapshot.get("TOTAL_EPOCHS", self.epochs_run)
        if self.config.epochs > saved_epochs:
            # Create fresh scheduler for remaining steps
            remaining_steps = (self.config.epochs - self.epochs_run) * len(self.train_loader)
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr,
                total_steps=remaining_steps,
                pct_start=0.2,
                anneal_strategy="linear",
            )
            logger.warning(
                f"Epochs extended ({saved_epochs} -> {self.config.epochs}). Created new LR scheduler for {remaining_steps} remaining steps."
            )
        else:
            self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])

        logger.info(f"Resuming training from epoch {self.epochs_run}")

    def _save_snapshot(self, epoch: int) -> None:
        """Save training state to a snapshot."""
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "SCHEDULER_STATE": self.scheduler.state_dict(),
            "EPOCHS_RUN": epoch + 1,
            "TOTAL_EPOCHS": self.config.epochs,
        }
        torch.save(snapshot, self.config.snapshot_path)
        logger.info(f"Epoch {epoch + 1} | Snapshot saved at {self.config.snapshot_path}")

    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> float:
        """Run a single training batch and return the loss."""
        x = source.to(self.device, non_blocking=True)
        y = targets.to(self.device, non_blocking=True)

        if self.device.type == "cuda":
            x = x.to(torch.bfloat16)
            y = y.to(torch.bfloat16)

        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss_fn(self.model(x), y)
        loss.backward()

        # Gradient clipping
        # This has to happen *after* loss.backward(), which automatically handles DDP
        # gradient averaging, but *before* optimizer.step(), which needs the clipped
        # gradients to update parameters.
        model: Any = self.model
        if self.world_size > 1:
            # Need to unwrap the module from the DDP wrapper here to get parameters.
            params = list(model.module.parameters())
        else:
            params = list(model.parameters())
        grads = [p.grad for p in params if p.grad is not None]
        grad_norm = get_total_norm(grads)
        clip_grads_with_norm_(params, self.config.max_grad_norm, grad_norm)

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def _run_epoch(self, epoch: int) -> float:
        """Run one training epoch and return total loss."""
        self.sampler.set_epoch(epoch)
        total_loss = 0.0

        b_sz = len(next(iter(self.train_loader))[0])
        if self.global_rank == 0:
            logger.info(f"Epoch {epoch + 1} | Batchsize: {b_sz} | Batches per epoch: {len(self.train_loader)}")

        for source, targets in self.train_loader:
            batch_loss = self._run_batch(source, targets)
            total_loss += batch_loss

        return total_loss

    def _sync_loss(self, total_loss: float) -> float:
        """Synchronize loss across all ranks via all_reduce."""
        if self.world_size <= 1:
            return total_loss

        loss_tensor = torch.tensor([total_loss], device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        return loss_tensor.item()

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases on rank 0 only."""
        if not self.config.use_wandb or self.global_rank != 0:
            return

        config_dict = {
            "max_grad_norm": self.config.max_grad_norm,
            "seed": self.config.seed,
            "lr": self.config.lr,
            "weight_decay": self.config.weight_decay,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "hidden_dim": self.config.hidden_dim,
            "use_compile": self.config.use_compile,
        }
        self.wandb_run = wandb.init(
            entity=self.config.wandb_entity,
            project=self.config.wandb_project,
            config=config_dict,
        )
        logger.info(f"W&B run: {self.wandb_run.name}")

    def _log_metrics(self, epoch: int, avg_loss: float, lr: float) -> None:
        """Log metrics to console and wandb."""
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, LR: {lr:.6f}")

        if self.wandb_run is not None:
            wandb.log(
                {"loss": avg_loss, "lr": lr, "epoch": epoch + 1},
                step=epoch + 1,
            )

    def train(self) -> None:  # noqa: C901
        """Run the full training loop."""
        self._init_wandb()

        model: Any = self.model
        if self.global_rank == 0:
            # Log model info
            if self.world_size > 1:
                params = model.module.parameters()
            else:
                params = model.parameters()
            num_params = sum(p.numel() for p in params)
            logger.info(f"Model: {num_params:,} params (~{num_params / 1e6:.1f}M)")

        # Always use bfloat16 on CUDA for efficiency
        if self.device.type == "cuda":
            if self.world_size > 1:
                model.module.to(torch.bfloat16)
            else:
                model.to(torch.bfloat16)

        # Check if training is already complete
        if self.epochs_run >= self.config.epochs:
            logger.warning(
                f"Training already complete ({self.epochs_run}/{self.config.epochs}"
                f" epochs). Use --epochs > {self.epochs_run} to continue training,"
                f" or delete {self.config.snapshot_path} to start fresh."
            )
            return

        for epoch in tqdm(
            range(self.epochs_run, self.config.epochs),
            disable=self.global_rank != 0,
        ):
            total_loss = self._run_epoch(epoch)
            total_loss = self._sync_loss(total_loss)

            if self.global_rank == 0:
                avg_loss = total_loss / len(self.train_loader)
                lr_now = self.optimizer.param_groups[0]["lr"]
                self._log_metrics(epoch, avg_loss, lr_now)

                # Save snapshot periodically
                if (epoch + 1) % self.config.save_every == 0:
                    self._save_snapshot(epoch)

        # Save final snapshot
        if self.global_rank == 0:
            self._save_snapshot(self.config.epochs - 1)

        # Finish wandb run
        if self.wandb_run is not None:
            wandb.finish()


def load_train_objs(
    config: TrainConfig,
    device: torch.device,
) -> tuple[TensorDataset, nn.Module]:
    """Load dataset and model."""
    # Create synthetic dataset
    x = torch.randn(10000, 10)
    y = torch.randn(10000, 1)
    dataset = TensorDataset(x, y)

    # Build model
    model = build_model(config.hidden_dim, device)

    return dataset, model


def create_optimizer(
    model: nn.Module,
    config: TrainConfig,
    device: torch.device,
    total_steps: int,
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    """Create AdamW optimizer with OneCycleLR scheduler."""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        fused=(device.type == "cuda"),
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.lr,
        total_steps=total_steps,
        pct_start=0.2,
        anneal_strategy="linear",
    )
    return optimizer, scheduler


def main(config: TrainConfig) -> None:
    """Main entry point for training."""
    seed_everything(config.seed)
    local_rank, global_rank, world_size, device = ddp_setup()

    # Configure dynamo before model creation
    configure_dynamo(config.use_compile, world_size)

    # Load dataset and model
    dataset, model = load_train_objs(config, device)

    # Prepare dataloader
    train_loader, sampler = prepare_dataloader(dataset, config.batch_size, world_size, global_rank, device)

    # Create optimizer and scheduler
    total_steps = config.epochs * len(train_loader)
    optimizer, scheduler = create_optimizer(model, config, device, total_steps)
    loss_fn = nn.MSELoss()

    # Create trainer and run
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        sampler=sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=config,
        device=device,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
    )
    trainer.train()

    # Cleanup
    dist.destroy_process_group()
    if global_rank == 0:
        print("Training complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distributed training with torchrun")
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs to train (default: 20)")
    parser.add_argument("--batch-size", type=int, default=256, help="Input batch size per device (default: 256)")
    parser.add_argument("--save-every", type=int, default=5, help="Save snapshot every N epochs (default: 5)")
    parser.add_argument("--snapshot-path", type=str, default="snapshot.pt", help="Path to save/load snapshots (default: snapshot.pt)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")

    args = parser.parse_args()

    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_every=args.save_every,
        snapshot_path=args.snapshot_path,
        lr=args.lr,
        use_compile=not args.no_compile,
        use_wandb=not args.no_wandb,
    )

    main(train_config)
