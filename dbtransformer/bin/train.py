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
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

import wandb
from dbtransformer.hardware_abstraction_layer import HardwareConfig
from dbtransformer.model import (
    MAX_F2P_NEIGHBORS,
    Batch,
    ModelOutput,
    RelationalTransformer,
)
from dbtransformer.profiling import (
    mps_profiler_context,
    no_profiler_context,
    torch_profiler_context,
    torch_profiler_full_context,
)


@dataclass
class TrainConfig:
    """Configuration for training."""

    # Training hyperparameters
    max_grad_norm: float = 1.0
    seed: int = 42
    lr: float = 0.01
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 8
    use_compile: bool = True
    save_every: int = 5
    snapshot_path: str = "snapshot.pt"

    # Model architecture.
    # Intentionally set small during testing and development.
    num_blocks: int = 2
    d_model: int = 128
    d_text: int = 384
    num_heads: int = 4
    d_ff: int = 4 * d_model
    seq_len: int = 512

    # Data loading
    num_workers: int = 0  # 0=sync, 1+=async background workers
    debug_timing: bool = False  # Enable per-phase timing in model forward

    # Dataset size for dummy data
    total_num_samples: int = 2048

    # Weights & Biases config
    use_wandb: bool = True
    wandb_entity: str = "mttrdmnd-massachusetts-institute-of-technology"
    wandb_project: str | None = "dbtransformer"

    # Profiling config
    profile_mode: str | None = None  # "torch", "mps", or None
    profile_output: str = "./profiler_logs"
    profile_wait: int = 2
    profile_warmup: int = 2
    profile_active: int = 6


def seed_everything(seed: int = 42) -> None:
    """Set seeds for reproducibility across all random sources."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddp_setup() -> tuple[int, int, int, HardwareConfig]:
    """
    Initialize distributed training.

    Returns:
        (local_rank, global_rank, world_size, device_config)
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ.get("RANK", "0"))

    # Auto-detect device configuration
    hardware_config = HardwareConfig.auto_detect(local_rank=local_rank)

    # Initialize process group with the appropriate backend
    dist.init_process_group(backend=hardware_config.ddp_backend)
    world_size = dist.get_world_size()

    # MPS doesn't support multi-process, fall back to CPU
    if hardware_config.is_mps and world_size > 1:
        logger.warning("MPS doesn't support multi-GPU; falling back to CPU")
        hardware_config = HardwareConfig.for_cpu()

    logger.info(
        f"[Local Rank {local_rank} | Global Rank {global_rank} | World Size {world_size}] "
        f"backend={hardware_config.ddp_backend}, device={hardware_config.device}"
    )
    hardware_config.log_config()

    return local_rank, global_rank, world_size, hardware_config


def build_model(
    config: TrainConfig,
    hardware_config: HardwareConfig,
) -> nn.Module:
    """Build the RelationalTransformer model."""
    return RelationalTransformer(
        num_blocks=config.num_blocks,
        d_model=config.d_model,
        d_text=config.d_text,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        hardware_config=hardware_config,
    ).to(hardware_config.device)


def create_dummy_batch(
    batch_size: int,
    seq_len: int,
    d_text: int,
    device: torch.device,
) -> Batch:
    """
    Create a dummy Batch with random data for testing.

    This generates synthetic data matching the Batch TypedDict structure
    expected by the RelationalTransformer model.
    """
    # Create node indices: groups of cells belong to the same row
    # We'll simulate ~10 cells per row on average
    cells_per_row = 10
    num_rows = (seq_len + cells_per_row - 1) // cells_per_row  # Round up
    # Create enough node indices to cover seq_len
    node_indices = torch.arange(num_rows, device=device)
    node_indices = node_indices.repeat_interleave(cells_per_row)[:seq_len]
    node_indices = node_indices.expand(batch_size, -1).contiguous()

    # Table and column indices: simulate ~5 tables, ~20 columns per table
    num_tables = 5
    num_cols_per_table = 20
    table_indices = torch.randint(0, num_tables, (batch_size, seq_len), device=device, dtype=torch.int32)
    column_indices = torch.randint(0, num_tables * num_cols_per_table, (batch_size, seq_len), device=device, dtype=torch.int32)

    # Foreign-to-primary neighbor indices
    # Each cell has up to MAX_F2P_NEIGHBORS parent row references
    # Use -1 for padding (no neighbor)
    f2p = torch.randint(-1, num_rows, (batch_size, seq_len, MAX_F2P_NEIGHBORS), device=device, dtype=torch.int32)

    # Numeric values (z-score normalized)
    # Use float32 here; will be converted to bfloat16 on CUDA in _move_batch
    number_vals = torch.randn(batch_size, seq_len, 1, device=device, dtype=torch.float32)
    datetime_vals = torch.randn(batch_size, seq_len, 1, device=device, dtype=torch.float32)
    boolean_vals = torch.randn(batch_size, seq_len, 1, device=device, dtype=torch.float32)

    # Text embeddings (pre-computed sentence transformer output)
    text_vals = torch.randn(batch_size, seq_len, d_text, device=device, dtype=torch.float32)
    column_name_vals = torch.randn(batch_size, seq_len, d_text, device=device, dtype=torch.float32)

    # Semantic types: 0=number, 1=text, 2=datetime, 3=boolean
    # Exclude text (1) from masked positions for now (model doesn't support)
    semantic_types = torch.randint(0, 4, (batch_size, seq_len), device=device, dtype=torch.long)

    # Masks: positions to hide and predict
    # Mask ~15% of non-text positions
    masks = torch.rand(batch_size, seq_len, device=device) < 0.15  # noqa: PLR2004
    # Don't mask text positions (model doesn't support text prediction yet)
    masks &= semantic_types != 1

    # Ensure at least one masked position per batch for loss computation
    for i in range(batch_size):
        if not masks[i].any():
            # Find a non-text position to mask
            non_text = (semantic_types[i] != 1).nonzero(as_tuple=True)[0]
            if len(non_text) > 0:
                masks[i, non_text[0]] = True

    # is_targets: same as masks for now
    is_targets = masks.clone()

    # is_task_nodes: first ~10% of rows are task table rows
    task_row_threshold = num_rows // 10
    is_task_nodes = node_indices < task_row_threshold

    # No padding in dummy data
    is_padding = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)

    # Class value indices (unused, set to -1)
    class_indices = torch.full((batch_size, seq_len), -1, device=device, dtype=torch.int32)

    return Batch(
        node_indices=node_indices.to(torch.int32),
        table_name_indices=table_indices,
        column_name_indices=column_indices,
        f2p_neighbor_indices=f2p,
        number_values=number_vals,
        datetime_values=datetime_vals,
        boolean_values=boolean_vals,
        text_values=text_vals,
        column_name_values=column_name_vals,
        semantic_types=semantic_types,
        masks=masks,
        is_targets=is_targets,
        is_task_nodes=is_task_nodes,
        is_padding=is_padding,
        class_value_indices=class_indices,
        true_batch_size=batch_size,
    )


class DummyBatchDataset(Dataset):
    """
    Dataset that pre-generates dummy Batch data for fast iteration.

    Pre-generates all batches at initialization to avoid slow on-the-fly
    random tensor generation during training.
    """

    def __init__(
        self,
        total_num_samples: int,
        batch_size: int,
        seq_len: int,
        d_text: int,
    ) -> None:
        self.total_num_samples = total_num_samples
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_text = d_text

        # Pre-generate a pool of batches to reuse (much faster than on-the-fly)
        # Use a smaller pool and cycle through it
        self._pool_size = min(64, total_num_samples)
        logger.info(f"Pre-generating {self._pool_size} dummy batches...")
        self._batch_pool = [
            create_dummy_batch(
                batch_size=1,
                seq_len=seq_len,
                d_text=d_text,
                device=torch.device("cpu"),
            )
            for _ in range(self._pool_size)
        ]
        logger.info("Done pre-generating batches.")

    def __len__(self) -> int:
        return self.total_num_samples

    def __getitem__(self, idx: int) -> Batch:
        # Return from pre-generated pool (cycling through)
        return self._batch_pool[idx % self._pool_size]  # type: ignore[index]


def collate_batches(batches: list[Batch]) -> Batch:
    """
    Collate multiple single-sample Batches into one batched Batch.

    Each input batch has shape (1, seq_len, ...), we concatenate along dim 0.
    """
    batch_size = len(batches)

    def cat_field(key: str) -> torch.Tensor:
        tensors = [b[key] for b in batches]  # type: ignore[literal-required]
        return torch.cat(tensors, dim=0)

    return Batch(
        node_indices=cat_field("node_indices"),
        table_name_indices=cat_field("table_name_indices"),
        column_name_indices=cat_field("column_name_indices"),
        f2p_neighbor_indices=cat_field("f2p_neighbor_indices"),
        number_values=cat_field("number_values"),
        datetime_values=cat_field("datetime_values"),
        boolean_values=cat_field("boolean_values"),
        text_values=cat_field("text_values"),
        column_name_values=cat_field("column_name_values"),
        semantic_types=cat_field("semantic_types"),
        masks=cat_field("masks"),
        is_targets=cat_field("is_targets"),
        is_task_nodes=cat_field("is_task_nodes"),
        is_padding=cat_field("is_padding"),
        class_value_indices=cat_field("class_value_indices"),
        true_batch_size=batch_size,
    )


class DeviceCollator:
    """
    Collate batches directly onto target device.

    For MPS: Creates tensors directly on device, leveraging unified memory.
    This eliminates the CPU->device "move" entirely since MPS can write
    directly to unified memory during collation.

    For CUDA: Still creates on CPU (with pinned memory via DataLoader),
    since async DMA transfer is more efficient than direct GPU allocation.

    Only usable with num_workers=0 (main process has device access).
    """

    def __init__(
        self,
        hardware_config: HardwareConfig,
    ) -> None:
        self.device = hardware_config.device
        self.float_dtype = hardware_config.model_dtype
        # Only place directly on device for MPS (unified memory benefit).
        # CUDA is better served by pinned memory + async transfer.
        self.use_direct_device = hardware_config.is_mps

    def __call__(self, batches: list[Batch]) -> Batch:
        batch_size = len(batches)

        if not self.use_direct_device:
            # CUDA/CPU: use standard collation (pinned memory handles it)
            return collate_batches(batches)

        # MPS: concatenate and place directly on device in one operation
        device = self.device
        float_dtype = self.float_dtype

        def cat_to_device(key: str) -> torch.Tensor:
            tensors = [b[key] for b in batches]  # type: ignore[literal-required]
            # torch.cat creates output on same device as inputs (CPU)
            # .to(device) with MPS unified memory is essentially a no-op
            # for the data, just creates the Metal buffer wrapper
            return torch.cat(tensors, dim=0).to(device)

        def cat_to_device_float(key: str) -> torch.Tensor:
            tensors = [b[key] for b in batches]  # type: ignore[literal-required]
            return torch.cat(tensors, dim=0).to(device, dtype=float_dtype)

        return Batch(
            node_indices=cat_to_device("node_indices"),
            table_name_indices=cat_to_device("table_name_indices"),
            column_name_indices=cat_to_device("column_name_indices"),
            f2p_neighbor_indices=cat_to_device("f2p_neighbor_indices"),
            number_values=cat_to_device_float("number_values"),
            datetime_values=cat_to_device_float("datetime_values"),
            boolean_values=cat_to_device_float("boolean_values"),
            text_values=cat_to_device_float("text_values"),
            column_name_values=cat_to_device_float("column_name_values"),
            semantic_types=cat_to_device("semantic_types"),
            masks=cat_to_device("masks"),
            is_targets=cat_to_device("is_targets"),
            is_task_nodes=cat_to_device("is_task_nodes"),
            is_padding=cat_to_device("is_padding"),
            class_value_indices=cat_to_device("class_value_indices"),
            true_batch_size=batch_size,
        )


def configure_dynamo(use_compile: bool, world_size: int) -> None:
    """Configure torch.compile / dynamo settings."""
    if not use_compile:
        return
    torch._dynamo.config.cache_size_limit = 64
    if world_size > 1:
        torch._dynamo.config.optimize_ddp = True


def prepare_dataloader(
    dataset: Dataset,
    batch_size: int,
    world_size: int,
    rank: int,
    hardware_config: HardwareConfig,
    num_workers: int = 0,
) -> tuple[DataLoader, DistributedSampler]:
    """Create distributed dataloader for the given dataset.

    Args:
        num_workers: Number of worker processes for data loading.
            0 = load in main process (synchronous, GPU waits for data)
            1 = one background worker (overlaps data prep with GPU compute)
            >1 = multiple workers (useful for heavy CPU preprocessing)

    For MPS with num_workers=0: Uses DeviceCollator to place batches
    directly on device during collation, eliminating the batch-to-device
    transfer step entirely (leverages unified memory architecture).
    """
    sampler: DistributedSampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    # Use device-aware collator when safe (num_workers=0 means main process
    # does collation, so we have device access). For MPS this eliminates
    # the batch-to-device overhead by collating directly into unified memory.
    collate_fn: Callable[[list[Batch]], Batch]
    if num_workers == 0:
        collate_fn = DeviceCollator(hardware_config)
    else:
        # Workers can't access GPU, must collate on CPU
        collate_fn = collate_batches

    loader: DataLoader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=hardware_config.pin_memory,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        # Persistent workers avoid re-spawning overhead between epochs
        persistent_workers=num_workers > 0,
        # Prefetch batches to overlap data loading with compute
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader, sampler


class Trainer:
    """Handles distributed training with checkpointing support."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        sampler: DistributedSampler,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        config: TrainConfig,
        hardware_config: HardwareConfig,
        local_rank: int,
        global_rank: int,
        world_size: int,
        profiler: torch.profiler.profile | None = None,
    ) -> None:
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.hardware_config = hardware_config
        self.config = config

        self.train_loader = train_loader
        self.sampler = sampler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.profiler = profiler

        self.epochs_run = 0
        self.wandb_run: wandb.sdk.wandb_run.Run | None = None

        # Wrap model with DDP
        self.model = self._wrap_model(model)

        # Optionally compile model (CUDA or MPS only)
        if config.use_compile and not hardware_config.is_cpu:
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

        if self.hardware_config.is_cuda:
            return DistributedDataParallel(model, device_ids=[self.local_rank])
        return DistributedDataParallel(model)

    def _load_snapshot(self, snapshot_path: str) -> None:
        """Load training state from a snapshot."""
        logger.info(f"Loading snapshot from {snapshot_path}")
        if self.hardware_config.is_cuda:
            loc = f"cuda:{self.local_rank}"
        else:
            loc = "cpu"
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

    def _move_batch_to_device(self, batch: Batch) -> Batch:
        """Move all tensors in a Batch to the target device.

        Optimizations:
        - non_blocking=True only for CUDA (MPS unified memory gets no benefit)
        - Skip move if tensor already on target device
        - Dtype conversion only when needed
        """
        device = self.hardware_config.device
        float_dtype = self.hardware_config.model_dtype
        # non_blocking only helps CUDA (async DMA); MPS is unified memory
        non_blocking = self.hardware_config.is_cuda
        # Compare device type, not device object (mps != mps:0)
        device_type = device.type

        def move(t: torch.Tensor) -> torch.Tensor:
            if t.device.type == device_type:
                return t
            return t.to(device, non_blocking=non_blocking)

        def move_float(t: torch.Tensor) -> torch.Tensor:
            if t.device.type == device_type and t.dtype == float_dtype:
                return t
            return t.to(device, dtype=float_dtype, non_blocking=non_blocking)

        return Batch(
            node_indices=move(batch["node_indices"]),
            table_name_indices=move(batch["table_name_indices"]),
            column_name_indices=move(batch["column_name_indices"]),
            f2p_neighbor_indices=move(batch["f2p_neighbor_indices"]),
            number_values=move_float(batch["number_values"]),
            datetime_values=move_float(batch["datetime_values"]),
            boolean_values=move_float(batch["boolean_values"]),
            text_values=move_float(batch["text_values"]),
            column_name_values=move_float(batch["column_name_values"]),
            semantic_types=move(batch["semantic_types"]),
            masks=move(batch["masks"]),
            is_targets=move(batch["is_targets"]),
            is_task_nodes=move(batch["is_task_nodes"]),
            is_padding=move(batch["is_padding"]),
            class_value_indices=move(batch["class_value_indices"]),
            true_batch_size=batch["true_batch_size"],
        )

    def _sync_device(self) -> None:
        """Explicit device sync for accurate timing."""
        if self.hardware_config.is_cuda:
            torch.cuda.synchronize()
        elif self.hardware_config.is_mps:
            torch.mps.synchronize()

    def _run_batch(self, batch: Batch) -> torch.Tensor:
        """Run a single training batch and return the loss (as a tensor)."""
        batch = self._move_batch_to_device(batch)

        self.optimizer.zero_grad(set_to_none=True)

        # Model returns ModelOutput with loss already computed
        output: ModelOutput = self.model(batch)
        loss = output["loss"]
        loss.backward()

        # Gradient clipping
        # This has to happen *after* loss.backward(), which automatically
        # handles DDP gradient averaging, but *before* optimizer.step(),
        # which needs the clipped gradients to update parameters.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()

        # Explicit sync to ensure GPU work completes before next batch.
        # This prevents sync time from "leaking" into data loading.
        # Without this, async GPU ops cause misleading profiler attribution.
        self._sync_device()

        # Notify profiler of step boundary (for torch.profiler schedule)
        if self.profiler is not None:
            self.profiler.step()

        # Return detached loss tensor to avoid holding computation graph.
        # We do NOT call .item() here to avoid a CUDA sync every batch.
        return loss.detach()

    def _run_epoch(self, epoch: int) -> torch.Tensor:
        """Run one training epoch and return total loss as a tensor."""
        self.sampler.set_epoch(epoch)
        # Accumulate on GPU to avoid per-batch CUDA syncs from .item()
        total_loss = torch.tensor(
            0.0,
            device=self.hardware_config.device,
            dtype=torch.float32,
        )

        for i, batch in enumerate(self.train_loader):
            if i == 0 and self.global_rank == 0:
                b_sz = batch["node_indices"].shape[0]
                logger.info(f"Epoch {epoch + 1} | Batchsize: {b_sz} samples | Batches per epoch: {len(self.train_loader)}")

            batch_loss = self._run_batch(batch)
            total_loss += batch_loss

        return total_loss

    def _sync_loss(self, total_loss: torch.Tensor) -> float:
        """Synchronize loss across all ranks via all_reduce and return scalar.

        This is the ONLY place we call .item() on the loss, ensuring we only
        trigger one CUDA sync per epoch instead of per batch.
        """
        if self.world_size > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)

        return total_loss.item()

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases on rank 0 only."""
        if not self.config.use_wandb or self.global_rank != 0:
            return

        config_dict = {
            # Training hyperparameters
            "max_grad_norm": self.config.max_grad_norm,
            "seed": self.config.seed,
            "lr": self.config.lr,
            "weight_decay": self.config.weight_decay,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "use_compile": self.config.use_compile,
            # Model architecture
            "num_blocks": self.config.num_blocks,
            "d_model": self.config.d_model,
            "d_text": self.config.d_text,
            "num_heads": self.config.num_heads,
            "d_ff": self.config.d_ff,
            "seq_len": self.config.seq_len,
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

    def _warmup_compile(self) -> None:
        """
        Run a full warmup training step to trigger all torch.compile graphs.

        This compiles forward pass, backward pass, gradient clipping, and
        optimizer step. We save and restore model state so warmup doesn't
        affect training.
        """
        if not self.config.use_compile or self.hardware_config.is_cpu:
            return

        logger.info("Running warmup step to trigger compilation...")
        start = time.time()

        # Save model state before warmup (clone to avoid reference issues)
        model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        warmup_batch: Batch = next(iter(self.train_loader))
        warmup_batch = self._move_batch_to_device(warmup_batch)

        # Run full training step to compile all graphs
        self.optimizer.zero_grad(set_to_none=True)
        output: ModelOutput = self.model(warmup_batch)
        loss = output["loss"]
        loss.backward()

        # Compile gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        # Compile optimizer step (fused kernels)
        self.optimizer.step()

        if self.hardware_config.is_cuda:
            torch.cuda.synchronize()
        elif self.hardware_config.is_mps:
            torch.mps.synchronize()

        # Restore model state (undo warmup weight updates)
        self.model.load_state_dict(model_state)

        # Reset optimizer state (clear momentum buffers from warmup)
        self.optimizer.state.clear()

        end = time.time()
        logger.success(f"Warmup complete, compilation finished in {end - start:.2f}s")

    def train(self) -> None:
        """Run the full training loop."""
        self._init_wandb()

        model: Any = self.model
        if self.global_rank == 0:
            if self.world_size > 1:
                params = model.module.parameters()
            else:
                params = model.parameters()
            num_params = sum(p.numel() for p in params)
            logger.info(f"Model: {num_params:,} params (~{num_params / 1e6:.1f}M)")

        model.to(self.hardware_config.model_dtype)

        self._warmup_compile()

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
            total_loss_scalar = self._sync_loss(total_loss)

            if self.global_rank == 0:
                avg_loss = total_loss_scalar / len(self.train_loader)
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
    hardware_config: HardwareConfig,
) -> tuple[Dataset, nn.Module]:
    """Load dataset and model."""
    # Create dummy batch dataset
    dataset: Dataset = DummyBatchDataset(
        total_num_samples=config.total_num_samples,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        d_text=config.d_text,
    )

    # Build model
    model = build_model(config, hardware_config)

    return dataset, model


def create_optimizer(
    model: nn.Module,
    config: TrainConfig,
    hardware_config: HardwareConfig,
    total_steps: int,
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    """Create AdamW optimizer with OneCycleLR scheduler."""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        fused=hardware_config.use_fused_optimizer,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.lr,
        total_steps=total_steps,
        pct_start=0.2,
        anneal_strategy="linear",
    )
    return optimizer, scheduler


def _run_training(
    config: TrainConfig,
    hardware_config: HardwareConfig,
    local_rank: int,
    global_rank: int,
    world_size: int,
    profiler: torch.profiler.profile | None = None,
) -> None:
    """Run the full training pipeline (setup + training)."""
    # Configure dynamo before model creation
    configure_dynamo(config.use_compile, world_size)

    # Load dataset and model
    dataset, model = load_train_objs(config, hardware_config)

    # Prepare dataloader
    train_loader, sampler = prepare_dataloader(
        dataset,
        config.batch_size,
        world_size,
        global_rank,
        hardware_config,
        num_workers=config.num_workers,
    )

    # Create optimizer and scheduler
    total_steps = config.epochs * len(train_loader)
    optimizer, scheduler = create_optimizer(model, config, hardware_config, total_steps)

    # Create trainer and run
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        sampler=sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        hardware_config=hardware_config,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        profiler=profiler,
    )
    trainer.train()


def main(config: TrainConfig) -> None:
    """Main entry point for training."""
    seed_everything(config.seed)
    local_rank, global_rank, world_size, hardware_config = ddp_setup()

    # Select profiler context based on config
    # "torch-full" wraps EVERYTHING (data loading, setup, training)
    # "torch" uses a step-based schedule (only profiles training batches)
    if config.profile_mode == "torch-full":
        profiler_ctx = torch_profiler_full_context(
            output_dir=config.profile_output,
        )
    elif config.profile_mode == "torch":
        profiler_ctx = torch_profiler_context(
            output_dir=config.profile_output,
            wait=config.profile_wait,
            warmup=config.profile_warmup,
            active=config.profile_active,
        )
    elif config.profile_mode == "mps":
        profiler_ctx = mps_profiler_context()
    else:
        profiler_ctx = no_profiler_context()

    # Run training (optionally with profiling)
    with profiler_ctx as prof:
        use_step_profiler = config.profile_mode == "torch"
        _run_training(
            config=config,
            hardware_config=hardware_config,
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size,
            profiler=prof if use_step_profiler else None,
        )

    # Cleanup
    dist.destroy_process_group()
    if global_rank == 0:
        print("Training complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distributed training with torchrun")

    # Training arguments
    parser.add_argument("--save-every", type=int, default=5, help="Save snapshot every N epochs (default: 5)")
    parser.add_argument("--snapshot-path", type=str, default="snapshot.pt", help="Path to save/load snapshots (default: snapshot.pt)")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0=sync, 1+=async, default: 0)")

    # Profiling arguments
    parser.add_argument(
        "--profile",
        type=str,
        choices=["torch", "torch-full", "mps"],
        default=None,
        help="Enable profiling: 'torch' for a few run_batch invocations (step-based), 'torch-full' for full program, 'mps' for Instruments",
    )
    parser.add_argument("--profile-output", type=str, default="./profiler_logs", help="Output dir for torch profiler (default: ./profiler_logs)")
    parser.add_argument("--profile-wait", type=int, default=2, help="Steps to skip before warmup (default: 2)")
    parser.add_argument("--profile-warmup", type=int, default=2, help="Warmup steps (default: 2)")
    parser.add_argument("--profile-active", type=int, default=6, help="Active profiling steps (default: 6)")

    args = parser.parse_args()

    train_config = TrainConfig(
        save_every=args.save_every,
        snapshot_path=args.snapshot_path,
        use_compile=not args.no_compile,
        use_wandb=not args.no_wandb,
        num_workers=args.num_workers,
        debug_timing=args.debug_timing,
        profile_mode=args.profile,
        profile_output=args.profile_output,
        profile_wait=args.profile_wait,
        profile_warmup=args.profile_warmup,
        profile_active=args.profile_active,
    )

    main(train_config)
