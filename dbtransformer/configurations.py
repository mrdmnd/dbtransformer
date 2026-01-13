# This file contains *all* of the various data classes that define the configurations
# for the different components of the system.


from dataclasses import dataclass
from typing import Literal

import torch


# This isn't really a "configuration" class - it's just a bunch of parameters that are
# set when we configure DDP.
@dataclass
class DDPParameters:
    local_rank: int  # Index of the GPU on the current node, always 0-7 basically.
    global_rank: int  # Index of the GPU on the entire cluster
    world_size: int  # Total number of GPUs in the cluster
    device: torch.device  # The specific device this global_rank points to.


##########################
# Training Configuration
##########################


@dataclass
class ModelConfig:
    """Model architecture hyperparameters.

    The model uses text embeddings for categoricals (pre-embedded as
    "<col_name> is <value>"), so no vocabulary size parameter is needed.
    This enables zero-shot transfer to new databases/categories.
    """

    model_dtype: torch.dtype = torch.bfloat16
    num_blocks: int = 12
    num_heads: int = 8
    d_model: int = 256
    d_text: int = 768  # BGE-base embedding dimension
    d_ff: int = 4 * d_model
    # Timestamp feature dimension: 5 cyclical components (sin/cos each) + 1 linear = 11
    # Cyclical: minute_of_hour, hour_of_day, day_of_week, day_of_year, month
    # Linear: epoch_seconds (z-scored, also used as prediction target)
    d_time: int = 11
    compile_model: bool = True


@dataclass
class TrainingConfig:
    # Hyperparameters
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0  # Used for gradient clipping
    seq_len: int = 1024
    batch_size: int = 32

    # Total requested batches to train for.
    max_batches: int = 300

    # Evaluation
    eval_every_n_batches: int = 100
    max_eval_batches: int = 16

    # Log metrics every N batches.
    log_every_n_batches: int = 10

    # Save a snapshot every N batches.
    save_every_n_batches: int = 100
    snapshot_path: str = "snapshot.pt"

    # DDP Configuration
    # 0=sync, 1+=async background workers. Should set to n_gpus probably.
    num_workers: int = 0
    ddp_backend: Literal["gloo", "nccl"] = "nccl"


@dataclass
class DataConfig:
    """
    Configuration for the training data.

    Supports multiple databases via two modes:
    1. Specify db_dir only: loads ALL database subdirectories from that directory
    2. Specify db_dir + db_names: loads only the named databases from db_dir

    """

    # Base directory containing preprocessed database subdirectories
    db_dir: str = "data/"
    # Optional list of database names to load (subdirectory names).
    # If None, all databases in db_dir are loaded.
    db_names: list[str] | None = None
    # Optional directory for eval databases (separate from training)
    eval_db_dir: str | None = None
    # Optional list of eval database names (if None, all in eval_db_dir)
    eval_db_names: list[str] | None = None


@dataclass
class SamplerConfig:
    """
    Configuration for the Rust sampler-backed dataset.

    Supports deterministic train/val/test splits via hash-based row assignment.
    Each row is assigned to a split based on hash(row_idx, split_seed), ensuring:
    - Reproducible splits across runs with the same split_seed
    - Works for ANY database regardless of schema
    - Approximately respects the configured fractions
    """

    max_bfs_width: int = 256
    seed: int = 42
    # Number of threads for parallel batch generation in Rust.
    # For multi-GPU training, set to num_cpus / world_size to avoid oversubscription.
    # If None, defaults to 1 thread per process (safe for multi-process training).
    num_threads: int | None = None

    # Random split configuration
    # Which split to sample seeds from ("train", "val", "test", or None for all)
    split: str | None = None
    # Fraction of rows for training (default 0.8)
    train_frac: float = 0.8
    # Fraction of rows for validation (default 0.1). Test gets the remainder.
    val_frac: float = 0.1
    # Seed for deterministic split assignment (separate from sampling seed)
    split_seed: int = 12345


@dataclass
class WandbConfig:
    """
    Configuration for the Weights & Biases logging system.
    """

    enabled: bool = True
    wandb_entity: str = "mttrdmnd-massachusetts-institute-of-technology"
    wandb_project: str = "dbtransformer"


@dataclass
class ProfilingConfig:
    """
    Configuration for the profiling system.
    """

    # "full" mode profiles everything (data loading, setup, training)
    # "batch" mode profiles only the training batches (default)
    # "full" mode generates huge profiles (O(10 GB) for a single run) because it is capturing EVERYTHING.
    # "batch" mode is much smaller (O(100 MB) for a single run) because it only captures
    # timing information for a few training batches.
    profile_output: str = "./profiler_logs"
    profile_mode: Literal["full", "batch", "disabled"] = "disabled"

    # If the profiler is in "batch" mode, we wait a few batches, warmup, then record for `active` batches.
    # Note: this requires the number of executed batches to be at least the sum of these.
    batch_profile_wait_batches: int = 2
    batch_profile_warmup_batches: int = 2
    batch_profile_active_batches: int = 6
    batch_profile_repeat_batches: int = 1


@dataclass
class OverallConfig:
    """
    Overall configuration for the training system.

    This is the top-level configuration object that is passed to the training script.
    """

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    sampler: SamplerConfig
    profiling: ProfilingConfig
    wandb: WandbConfig
    random_seed: int = 42069


# Default configuration instance
DEFAULT_OVERALL_CONFIG = OverallConfig(
    data=DataConfig(
        db_dir="/home/mrdmnd/data/databases_preprocessed",
        db_names=["synthetic-xor", "synthetic-moons", "synthetic-numerical", "synthetic-teams"],
    ),
    model=ModelConfig(),
    training=TrainingConfig(
        batch_size=32,
        seq_len=1024,
        max_batches=500,
        eval_every_n_batches=50,
        log_every_n_batches=10,
        save_every_n_batches=500,
        num_workers=4,  # DataLoader workers for prefetching
    ),
    sampler=SamplerConfig(
        num_threads=32,  # Per-worker threads (4 workers × 6 = 24 total)
        split="train",  # Enable train/val splitting
        train_frac=0.8,
        val_frac=0.1,
        split_seed=12345,
    ),
    profiling=ProfilingConfig(profile_mode="disabled"),
    wandb=WandbConfig(enabled=True),
)
