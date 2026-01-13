"""
PyTorch Distributed Training Script
====================================
Always run with uv run torchrun:
    Single process:   uv run torchrun --nproc_per_node=1 dbtransformer/train.py --num-workers 1
    2 GPUs:           uv run torchrun --nproc_per_node=2 dbtransformer/train.py --num-workers 2
    8 GPUs:           uv run torchrun --nproc_per_node=8 dbtransformer/train.py --num-workers 8
"""

import dataclasses
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
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.sdk.wandb_run import Run as WandbRun

import wandb
from dbtransformer.configurations import (
    DEFAULT_OVERALL_CONFIG,
    DataConfig,
    DDPParameters,
    OverallConfig,
)
from dbtransformer.model import (
    Batch,
    ModelOutput,
    RelationalTransformer,
    SemanticType,
)
from dbtransformer.profiling import (
    get_profiler_context,
)
from dbtransformer.sampler_dataset import SamplerBatchDataset

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires CUDA.")


def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddp_setup(backend: Literal["gloo", "nccl"]) -> DDPParameters:
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
    logger.warning("Cleaning up DDP process group.")
    dist.destroy_process_group()


class Trainer:
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
        self.wandb_run: WandbRun | None = None

        self.model: nn.Module = RelationalTransformer(config.model)
        self.model.to(
            device=self.ddp_parameters.device,
            dtype=self.config.model.model_dtype,
        )
        params = self.model.parameters()
        num_params = sum(p.numel() for p in params)
        if self.is_leader:
            logger.info(f"Model: {num_params:,} params (~{num_params / 1e6:.1f}M)")

        # Training dataset: use split="train" if splitting is enabled
        train_sampler_config = dataclasses.replace(
            config.sampler,
            split="train" if config.sampler.split is not None else None,
        )
        self.dataset = SamplerBatchDataset(
            data_config=config.data,
            model_config=config.model,
            training_config=config.training,
            sampler_config=train_sampler_config,
            ddp_parameters=self.ddp_parameters,
        )

        # Rust sampler already partitions by rank/world_size internally so we DO NOT
        # need to wrap it with a DistributedSampler.
        self.dist_sampler = None  # Not needed; sharding handled in SamplerBatchDataset
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=None,
            num_workers=config.training.num_workers,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            persistent_workers=config.training.num_workers > 0,
            prefetch_factor=2 if config.training.num_workers > 0 else None,
        )

        # Eval dataset: use split="val" on the same databases
        # This uses the same train/val/test fractions and split_seed for consistency
        self.eval_dataset: SamplerBatchDataset | None = None
        self.eval_dataloader: DataLoader[Batch] | None = None

        # Create eval dataset if splitting is enabled OR if a separate eval_db_dir is specified
        if config.sampler.split is not None or config.data.eval_db_dir is not None:
            eval_data_config = DataConfig(
                db_dir=config.data.eval_db_dir or config.data.db_dir,
                db_names=config.data.eval_db_names or config.data.db_names,
            )
            eval_sampler_config = dataclasses.replace(
                config.sampler,
                split="val",  # Always use validation split for eval
            )
            self.eval_dataset = SamplerBatchDataset(
                data_config=eval_data_config,
                model_config=config.model,
                training_config=config.training,
                sampler_config=eval_sampler_config,
                ddp_parameters=self.ddp_parameters,
            )
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=None,
                num_workers=0,
                pin_memory=True,
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

        # Compile model unless profiling (record_function annotations inside
        # the compiled forward pass get eliminated, so we skip compile for
        # detailed profiling)
        if config.profiling.profile_mode == "disabled":
            if self.is_leader:
                logger.info("Compiling model")
            self.model = torch.compile(self.model, dynamic=False)  # type: ignore[assignment]
        elif self.is_leader:
            logger.info("Skipping model compilation (profiling enabled)")

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
        with torch.autograd.profiler.record_function("batch_to_device"):
            batch.to_device(
                self.ddp_parameters.device,
                float_dtype=self.config.model.model_dtype,
            )

        with torch.autograd.profiler.record_function("optimizer_zero_grad"):
            self.optimizer.zero_grad(set_to_none=True)

        # Model returns ModelOutput with loss already computed
        with torch.autograd.profiler.record_function("forward_pass"):
            output: ModelOutput = self.model(batch)
            loss = output["loss"]

        with torch.autograd.profiler.record_function("backward_pass"):
            loss.backward()

        # Gradient clipping
        # This has to happen *after* loss.backward(), which automatically
        # handles DDP gradient averaging, but *before* optimizer.step(),
        # which needs the clipped gradients to update parameters.
        with torch.autograd.profiler.record_function("gradient_clipping"):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)

        with torch.autograd.profiler.record_function("optimizer_step"):
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
            if self.dist_sampler is not None:
                self.dist_sampler.set_epoch(self.current_epoch)
            if hasattr(self.dataset, "set_epoch"):
                self.dataset.set_epoch(self.current_epoch)
            yield from self.dataloader
            self.current_epoch += 1

    def _gather_varlen(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather variable-length 1D tensors across ranks and concatenate."""
        if not dist.is_initialized():
            return tensor

        local_len = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.int64)
        lengths = [torch.zeros_like(local_len) for _ in range(dist.get_world_size())]
        dist.all_gather(lengths, local_len)
        max_len = int(torch.stack(lengths).max())

        if tensor.shape[0] < max_len:
            pad = torch.zeros(max_len - tensor.shape[0], device=tensor.device, dtype=tensor.dtype)
            tensor_padded = torch.cat([tensor, pad], dim=0)
        else:
            tensor_padded = tensor

        gathered = [torch.zeros_like(tensor_padded) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor_padded)

        trimmed = []
        for t, l in zip(gathered, lengths):
            trimmed.append(t[: int(l.item())])
        return torch.cat(trimmed, dim=0)

    def _evaluate(self) -> dict[str, float]:
        """Run a bounded evaluation loop over the eval dataloader."""
        if self.eval_dataloader is None:
            return {}

        # Fixed epoch for eval to keep ordering deterministic
        if hasattr(self.eval_dataset, "set_epoch"):
            self.eval_dataset.set_epoch(0)

        self.model.eval()

        device = self.ddp_parameters.device
        float_dtype = self.config.model.model_dtype

        loss_sum = torch.zeros([], device=device, dtype=torch.float32)
        batch_count = torch.zeros([], device=device, dtype=torch.float32)

        # Collect predictions/labels by semantic type
        numerical_preds_list: list[torch.Tensor] = []
        numerical_labels_list: list[torch.Tensor] = []
        categorical_preds_list: list[torch.Tensor] = []  # cosine similarities
        categorical_labels_list: list[torch.Tensor] = []  # 1s (correct match)
        timestamp_preds_list: list[torch.Tensor] = []
        timestamp_labels_list: list[torch.Tensor] = []

        max_batches = self.config.training.max_eval_batches

        with torch.inference_mode():
            for i, batch in enumerate(self.eval_dataloader):
                if i >= max_batches:
                    break

                batch.to_device(device, float_dtype=float_dtype)
                output: ModelOutput = self.model(batch)

                loss_sum += output["loss"].detach()
                batch_count += 1.0

                mask_active = batch.masks & (~batch.is_padding)
                semantic = batch.semantic_types

                # Numerical: regression metrics (R², MAE)
                num_mask = mask_active & (semantic == SemanticType.NUMERICAL.value)
                if num_mask.any() and output["yhat_numerical"] is not None:
                    preds = output["yhat_numerical"][num_mask].flatten()
                    labels = batch.numerical_values[num_mask].flatten()
                    numerical_preds_list.append(preds.detach())
                    numerical_labels_list.append(labels.detach())

                # Categorical: cosine similarity (higher = better match)
                cat_mask = mask_active & (semantic == SemanticType.CATEGORICAL.value)
                if cat_mask.any() and output["yhat_categorical"] is not None:
                    pred_emb = output["yhat_categorical"][cat_mask]
                    target_emb = batch.categorical_values[cat_mask]
                    # Compute cosine similarity per sample
                    cos_sim = torch.nn.functional.cosine_similarity(pred_emb, target_emb, dim=-1)
                    categorical_preds_list.append(cos_sim.detach())
                    categorical_labels_list.append(torch.ones_like(cos_sim))  # target is 1.0

                # Timestamp: regression on z-scored epoch seconds
                ts_mask = mask_active & (semantic == SemanticType.TIMESTAMP.value)
                if ts_mask.any() and output["yhat_timestamp"] is not None:
                    preds = output["yhat_timestamp"][ts_mask].flatten()
                    # Target is the last component of timestamp_values (z-scored epoch)
                    labels = batch.timestamp_values[ts_mask][..., -1].flatten()
                    timestamp_preds_list.append(preds.detach())
                    timestamp_labels_list.append(labels.detach())

        # Reduce loss totals
        if dist.is_initialized():
            for tensor in [loss_sum, batch_count]:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        metrics: dict[str, float] = {}
        if batch_count.item() > 0:
            metrics["eval/loss"] = (loss_sum / batch_count).item()

        # Numerical metrics: R² and MAE
        if numerical_preds_list:
            num_preds = torch.cat(numerical_preds_list, dim=0)
            num_labels = torch.cat(numerical_labels_list, dim=0)
            num_preds = self._gather_varlen(num_preds)
            num_labels = self._gather_varlen(num_labels)

            if self.is_leader and num_labels.numel() > 0:
                y = num_labels.float().cpu().numpy()
                yhat = num_preds.float().cpu().numpy()
                # R²: 1 - SS_res / SS_tot
                ss_res = float(np.sum((y - yhat) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2)) if y.size > 0 else 0.0
                if ss_tot > 0:
                    metrics["eval/r2_numerical"] = 1.0 - ss_res / ss_tot
                metrics["eval/mae_numerical"] = float(np.mean(np.abs(y - yhat)))

        # Categorical metrics: mean cosine similarity (should approach 1.0)
        if categorical_preds_list:
            cat_preds = torch.cat(categorical_preds_list, dim=0)
            cat_preds = self._gather_varlen(cat_preds)

            if self.is_leader and cat_preds.numel() > 0:
                metrics["eval/mean_cos_sim_categorical"] = float(cat_preds.mean().item())

        # Timestamp metrics: R² and MAE (same as numerical)
        if timestamp_preds_list:
            ts_preds = torch.cat(timestamp_preds_list, dim=0)
            ts_labels = torch.cat(timestamp_labels_list, dim=0)
            ts_preds = self._gather_varlen(ts_preds)
            ts_labels = self._gather_varlen(ts_labels)

            if self.is_leader and ts_labels.numel() > 0:
                y = ts_labels.float().cpu().numpy()
                yhat = ts_preds.float().cpu().numpy()
                ss_res = float(np.sum((y - yhat) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2)) if y.size > 0 else 0.0
                if ss_tot > 0:
                    metrics["eval/r2_timestamp"] = 1.0 - ss_res / ss_tot
                metrics["eval/mae_timestamp"] = float(np.mean(np.abs(y - yhat)))

        if self.is_leader and metrics:
            logger.info("Eval metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if self.wandb_run is not None:
                self.wandb_run.log(metrics, step=self.batches_run)

        self.model.train()
        return metrics

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases on rank 0 only."""

    def _warmup_compile(self) -> None:
        """
        Run a full warmup training step to trigger all torch.compile graphs.

        This compiles forward pass, backward pass, gradient clipping, and
        optimizer step. We save and restore model state so warmup doesn't
        affect training.
        """
        if self.is_leader:
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
        if self.is_leader:
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
            with torch.autograd.profiler.record_function("data_loading"):
                batch = next(batch_iter)
            batch_loss = self._run_batch(batch)
            accumulated_loss += batch_loss
            batches_since_log += 1
            current_batch = batch_num + 1

            if (
                self.eval_dataloader is not None
                and self.config.training.eval_every_n_batches > 0
                and current_batch % self.config.training.eval_every_n_batches == 0
            ):
                self._evaluate()

            # Log metrics every log_every_n_batches
            if current_batch % self.config.training.log_every_n_batches == 0:
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
        samples_per_sec = (batches_trained * self.config.training.batch_size * self.ddp_parameters.world_size) / elapsed
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
        logger.success(f"Starting training with config:\n{config!r}")
    profiler_ctx = get_profiler_context(config.profiling)

    with profiler_ctx as prof:
        # Configure torch before model creation
        # torch.set_float32_matmul_precision("high")
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.optimize_ddp = True
        # torch.set_num_threads(1)

        trainer = Trainer(config=config, ddp_parameters=ddp_parameters, profiler=prof)
        trainer.train()

    ddp_cleanup()


if __name__ == "__main__":
    # Use baked-in dataclass defaults; customize by editing DEFAULT_OVERALL_CONFIG or
    # import and call main(config) yourself.
    main(DEFAULT_OVERALL_CONFIG)
