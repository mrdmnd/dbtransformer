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


# This isn't part of the training system; but it's used for data preprocessing.
@dataclass
class EmbedderConfig:
    embedder_model_dtype: torch.dtype = torch.bfloat16
    embedder_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    compile_embedder: bool = True

    # Input strings longer than this are truncated.
    max_length: int = 1024

    # MUST MATCH d_text in ModelConfig!
    mrl_dimension: int = 384


##########################
# Training Configuration
##########################


@dataclass
class ModelConfig:
    model_dtype: torch.dtype = torch.bfloat16
    num_blocks: int = 12
    num_heads: int = 8
    d_model: int = 256
    d_text: int = 384
    d_ff: int = 4 * d_model
    seq_len: int = 1024
    batch_size: int = 32
    compile_model: bool = True


@dataclass
class TrainingConfig:
    # Hyperparameters
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0  # Gradient clipping

    # Total requested batches to train for.
    max_batches: int = 300

    # Synchronization and saving

    # Synchronize the loss across all ranks every N batches.
    sync_every_n_batches: int = 10

    # Save a snapshot every N batches.
    save_every_n_batches: int = 100
    snapshot_path: str = "snapshot.pt"

    # DDP Configuration
    # 0=sync, 1+=async background workers. Should set to n_gpus probably.
    num_workers: int = 0
    ddp_backend: Literal["gloo", "nccl"] = "nccl"


@dataclass
class DummyDataConfig:
    # Our "full dummy dataset" consists of 8192 samples (of sequences of length `seq_len`)
    total_num_samples: int = 8192


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
    # Shapes, memory, stack traces, etc.
    # "batch" mode is much smaller (O(100 MB) for a single run) because it only captures
    # timing information for a few training batches.
    profile_output: str = "./profiler_logs"
    profile_mode: Literal["full", "batch", "disabled"] = "disabled"

    # If the profiler is in "batch" mode, we wait for a few batches, warmup for a few,
    # and then record for `active` batches.
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

    model: ModelConfig
    training: TrainingConfig
    data: DummyDataConfig
    profiling: ProfilingConfig
    wandb: WandbConfig
    random_seed: int = 42069
