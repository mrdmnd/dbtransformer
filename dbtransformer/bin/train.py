"""
PyTorch Distributed Training Script
====================================
Always run with uv run torchrun:
    Single process:   uv run torchrun --nproc_per_node=1 dbtransformer/bin/train.py --num-workers 1
    2 GPUs:           uv run torchrun --nproc_per_node=2 dbtransformer/bin/train.py --num-workers 2
    8 GPUs:           uv run torchrun --nproc_per_node=8 dbtransformer/bin/train.py --num-workers 8
"""

import os
import random
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb
from dbtransformer.configurations import DDPParameters, DummyDataConfig, ModelConfig, OverallConfig, ProfilingConfig, TrainingConfig, WandbConfig
from dbtransformer.dummy_data import DummySampleDataset, collate_samples
from dbtransformer.model import (
    Batch,
    ModelOutput,
    RelationalTransformer,
)
from dbtransformer.profiling import (
    get_profiler_context,
)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires CUDA.")


def seed_everything(seed: int = 42) -> None:
    """Set seeds for reproducibility across all random sources."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddp_setup(backend: Literal["gloo", "nccl"]) -> DDPParameters:
    """
    Initialize distributed training.

    Handles multi-node (in theory?) and single-node multi-gpu training.
    """
    local_rank: int = int(os.environ["LOCAL_RANK"])
    global_rank: int = int(os.environ["RANK"])
    device: torch.device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend=backend)
    world_size: int = dist.get_world_size()

    # Each process prints it's own config line
    logger.info(f"[Local Rank {local_rank} | Global Rank {global_rank} | World Size {world_size}]")

    return DDPParameters(
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        device=device,
    )


def ddp_cleanup() -> None:
    """
    Clean up distributed training.
    """
    logger.warning("Cleaning up DDP process group.")
    dist.destroy_process_group()


class Trainer:
    """
    Handles distributed training with checkpointing support.
    """

    def __init__(
        self,
        config: OverallConfig,
        ddp_parameters: DDPParameters,
        profiler: torch.profiler.profile | None,
    ) -> None:
        self.config = config
        self.ddp_parameters = ddp_parameters
        self.is_leader: bool = self.ddp_parameters.global_rank == 0
        self.profiler = profiler

        self.batches_run = 0
        self.current_epoch = 0  # For sampler shuffling
        self.wandb_run: wandb.sdk.wandb_run.Run | None = None

        self.model: nn.Module = RelationalTransformer(config.model)
        self.model.to(
            device=self.ddp_parameters.device,
            dtype=self.config.model.model_dtype,
        )
        params = self.model.parameters()
        num_params = sum(p.numel() for p in params)
        if self.is_leader:
            logger.info(f"Model: {num_params:,} params (~{num_params / 1e6:.1f}M)")

        self.dataset = DummySampleDataset(config.data, config.model)
        self.sampler: DistributedSampler = DistributedSampler(
            self.dataset,
            num_replicas=self.ddp_parameters.world_size,
            rank=self.ddp_parameters.global_rank,
            shuffle=True,
            drop_last=False,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.model.batch_size,
            num_workers=config.training.num_workers,
            pin_memory=True,
            sampler=self.sampler,
            collate_fn=collate_samples,
            persistent_workers=config.training.num_workers > 0,
            prefetch_factor=2 if config.training.num_workers > 0 else None,
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            fused=True,
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.training.learning_rate,
            total_steps=config.training.max_batches,
            pct_start=0.2,
            anneal_strategy="linear",
        )
        if self.is_leader:
            logger.info("Wrapping model with DDP")
        self.model = DistributedDataParallel(self.model, device_ids=[self.ddp_parameters.local_rank])

        if self.is_leader:
            logger.info("Compiling model")
        self.model = torch.compile(self.model, dynamic=False)  # type: ignore[assignment]

        # Load snapshot after DDP/compile so state dict keys match
        if Path(config.training.snapshot_path).exists():
            self._load_snapshot(config.training.snapshot_path)

    def _load_snapshot(self, snapshot_path: str) -> None:
        """Load training state from a snapshot."""
        if self.is_leader:
            logger.info(f"Loading snapshot from {snapshot_path}")
        snapshot = torch.load(snapshot_path, map_location=f"cuda:{self.ddp_parameters.local_rank}")
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])

        # Cast optimizer state to match model dtype for fused optimizer
        # compatibility when resuming with different precision settings.
        model_dtype = self.config.model.model_dtype
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    state[k] = v.to(dtype=model_dtype)

        self.batches_run = snapshot["BATCHES_RUN"]
        self.current_epoch = snapshot.get("CURRENT_EPOCH", 0)

        # Check if max_batches was extended beyond original training
        saved_max_batches = snapshot.get("MAX_BATCHES", self.batches_run)
        if self.config.training.max_batches > saved_max_batches:
            # Create fresh scheduler for remaining steps
            remaining_steps = self.config.training.max_batches - self.batches_run
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.training.learning_rate,
                total_steps=remaining_steps,
                pct_start=0.2,
                anneal_strategy="linear",
            )
            if self.is_leader:
                logger.warning(
                    f"Max batches extended ({saved_max_batches} -> "
                    f"{self.config.training.max_batches}). Created new LR scheduler "
                    f"for {remaining_steps} remaining steps."
                )
        else:
            self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])

        if self.is_leader:
            logger.info(f"Resuming training from batch {self.batches_run}")

    def _save_snapshot(self, batch_num: int) -> None:
        """Save training state to a snapshot."""
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "SCHEDULER_STATE": self.scheduler.state_dict(),
            "BATCHES_RUN": batch_num,
            "CURRENT_EPOCH": self.current_epoch,
            "MAX_BATCHES": self.config.training.max_batches,
        }
        torch.save(snapshot, self.config.training.snapshot_path)
        logger.info(f"Batch {batch_num} | Snapshot saved at {self.config.training.snapshot_path}")

    def _run_batch(self, batch: Batch) -> torch.Tensor:
        """Run a single training batch and return the loss (as a tensor)."""
        batch.to_device(
            self.ddp_parameters.device,
            float_dtype=self.config.model.model_dtype,
        )

        self.optimizer.zero_grad(set_to_none=True)

        # Model returns ModelOutput with loss already computed
        output: ModelOutput = self.model(batch)
        loss = output["loss"]
        loss.backward()

        # Gradient clipping
        # This has to happen *after* loss.backward(), which automatically
        # handles DDP gradient averaging, but *before* optimizer.step(),
        # which needs the clipped gradients to update parameters.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()

        # Explicit sync to ensure GPU work completes before next batch.
        # This prevents sync time from "leaking" into data loading.
        # Without this, async GPU ops cause misleading profiler attribution.

        # Notify profiler of step boundary (for torch.profiler schedule)
        if self.profiler is not None:
            torch.cuda.synchronize()
            self.profiler.step()

        # Return detached loss tensor to avoid holding computation graph.
        # We do NOT call .item() here to avoid a CUDA sync every batch.
        return loss.detach()

    def _batch_iterator(self) -> Iterator[Batch]:
        """
        Yield batches infinitely, cycling through epochs.

        Updates sampler.set_epoch() on each new epoch for proper shuffling
        in distributed training.
        """
        while True:
            self.sampler.set_epoch(self.current_epoch)
            yield from self.dataloader
            self.current_epoch += 1

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases on rank 0 only."""

    def _warmup_compile(self) -> None:
        """
        Run a full warmup training step to trigger all torch.compile graphs.

        This compiles forward pass, backward pass, gradient clipping, and
        optimizer step. We save and restore model state so warmup doesn't
        affect training.
        """
        logger.info("Running warmup step to trigger compilation...")
        start = time.time()

        # Save model state before warmup (clone to avoid reference issues)
        model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        warmup_batch: Batch = next(iter(self.dataloader))
        warmup_batch.to_device(
            self.ddp_parameters.device,
            float_dtype=self.config.model.model_dtype,
        )

        # Run full training step to compile all graphs
        self.optimizer.zero_grad(set_to_none=True)
        output: ModelOutput = self.model(warmup_batch)
        loss = output["loss"]
        loss.backward()
        # Compile gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
        # Compile optimizer step (fused kernels)
        self.optimizer.step()
        # Synchronize to ensure warmup is complete
        torch.cuda.synchronize()
        # Restore model state (undo warmup weight updates)
        self.model.load_state_dict(model_state)
        # Reset optimizer state (clear momentum buffers from warmup)
        self.optimizer.state.clear()
        end = time.time()
        logger.success(f"Warmup complete, compilation finished in {end - start:.2f}s")

    def train(self) -> None:
        """Run the full training loop."""
        # Initialize W&B on rank 0 only
        if self.config.wandb.enabled and self.is_leader:
            self.wandb_run = wandb.init(
                entity=self.config.wandb.wandb_entity,
                project=self.config.wandb.wandb_project,
                config=asdict(self.config),
            )
            logger.info(f"W&B run: {self.wandb_run.name}")

        self._warmup_compile()

        if self.batches_run >= self.config.training.max_batches:
            logger.warning(
                f"Training already complete ({self.batches_run}/"
                f"{self.config.training.max_batches} batches). "
                f"Use --max-batches > {self.batches_run} to continue training, "
                f"or delete {self.config.training.snapshot_path} to start fresh."
            )
            return

        # Accumulate loss over log_every_batches for averaging
        accumulated_loss = torch.tensor(
            0.0,
            device=self.ddp_parameters.device,
            dtype=torch.float32,
        )
        batches_since_log = 0
        batch_iter = self._batch_iterator()

        start_time = time.time()
        pbar = tqdm(
            range(self.batches_run, self.config.training.max_batches),
            initial=self.batches_run,
            total=self.config.training.max_batches,
            disable=self.ddp_parameters.global_rank != 0,
            desc="Training",
            unit="batch",
        )

        for batch_num in pbar:
            batch = next(batch_iter)
            batch_loss = self._run_batch(batch)
            accumulated_loss += batch_loss
            batches_since_log += 1
            current_batch = batch_num + 1

            # Sync and log metrics every sync_every_n_batches
            if current_batch % self.config.training.sync_every_n_batches == 0:
                if self.ddp_parameters.world_size > 1:
                    dist.all_reduce(accumulated_loss, op=dist.ReduceOp.AVG)

                total_loss_scalar = accumulated_loss.item()

                if self.is_leader:
                    avg_loss = total_loss_scalar / batches_since_log
                    lr_now = self.optimizer.param_groups[0]["lr"]
                    _log_metrics(self.wandb_run, current_batch, self.config.training.max_batches, avg_loss, lr_now)
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}")
                # Reset accumulator
                accumulated_loss = torch.tensor(
                    0.0,
                    device=self.ddp_parameters.device,
                    dtype=torch.float32,
                )
                batches_since_log = 0

            # Save snapshot every save_every_batches
            if current_batch % self.config.training.save_every_n_batches == 0 and self.ddp_parameters.global_rank == 0:
                self._save_snapshot(current_batch)

        elapsed = time.time() - start_time
        batches_trained = self.config.training.max_batches - self.batches_run
        samples_per_sec = (batches_trained * self.config.model.batch_size * self.ddp_parameters.world_size) / elapsed
        if self.is_leader:
            logger.info(f"Global throughput: {samples_per_sec:.1f} samples/sec")

        # Save final snapshot
        if self.is_leader:
            self._save_snapshot(self.config.training.max_batches)

        # Finish wandb run
        if self.is_leader and self.wandb_run is not None:
            wandb.finish()


def _log_metrics(
    wandb_run: wandb.sdk.wandb_run.Run | None,
    batch_num: int,
    max_batches: int,
    avg_loss: float,
    lr: float,
) -> None:
    """Log metrics to console and wandb."""
    logger.info(f"Batch {batch_num}/{max_batches}, Loss: {avg_loss:.4f}, LR: {lr:.6f}")
    if wandb_run is not None:
        wandb_run.log({"loss": avg_loss, "lr": lr, "batch": batch_num}, step=batch_num)


def main(config: OverallConfig) -> None:
    """Main entry point for training."""
    seed_everything(config.random_seed)
    ddp_parameters: DDPParameters = ddp_setup(config.training.ddp_backend)
    if ddp_parameters.global_rank == 0:
        logger.success(f"Starting training with config:\n{overall_config!r}")
    profiler_ctx = get_profiler_context(config.profiling)

    with profiler_ctx as prof:
        # Configure torch before model creation
        torch.set_float32_matmul_precision("high")
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.optimize_ddp = True
        torch.set_num_threads(1)

        trainer = Trainer(config=config, ddp_parameters=ddp_parameters, profiler=prof)
        trainer.train()

    ddp_cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distributed training with torchrun")

    # Training arguments
    parser.add_argument(
        "--snapshot-path",
        type=str,
        default="snapshot.pt",
        help="Path to save/load snapshots (default: snapshot.pt)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0=sync, 1+=async, default: 0)",
    )
    # Profiling arguments
    parser.add_argument(
        "--profile",
        type=str,
        choices=["full", "batch"],
        default=None,
        help="Enable profiling: 'full' for everything, 'batch' for a few run_batch invocations (step-based)",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default="./profiler_logs",
        help="Output dir for torch profiler (default: ./profiler_logs)",
    )

    args = parser.parse_args()

    wandb_config = WandbConfig()
    if args.no_wandb:
        wandb_config.enabled = False
    profile_config = ProfilingConfig()
    if args.profile is not None:
        profile_config.profile_mode = args.profile
    if args.profile_output is not None:
        profile_config.profile_output = args.profile_output

    overall_config = OverallConfig(
        model=ModelConfig(),
        training=TrainingConfig(),
        data=DummyDataConfig(),
        profiling=profile_config,
        wandb=wandb_config,
    )

    main(overall_config)
