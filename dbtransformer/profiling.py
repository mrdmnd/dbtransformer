"""
Profiling utilities for PyTorch training.

Usage with train.py:
    # Profile ONLY the training batches with torch.profiler (step-based)
    uv run torchrun --nproc_per_node=1 dbtransformer/bin/train.py \
        --profile batch --no-wandb

    # Profile EVERYTHING (data loading, setup, compilation, training)
    uv run torchrun --nproc_per_node=1 dbtransformer/bin/train.py \
        --profile full --no-wandb

    # The full profile will be VERY large (O(10 GB)) because it captures
    # EVERYTHING: shapes, memory, stack traces, etc.

    # View profiles with perfetto (ui.perfetto.dev) or TensorBoard.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import torch
from loguru import logger

from dbtransformer.configurations import ProfilingConfig


@contextmanager
def _batch_context(
    config: ProfilingConfig,
) -> Iterator[torch.profiler.profile]:
    """
    Context manager for torch.profiler with step-based schedule.

    Uses a schedule to wait, warmup, then record for a configurable
    number of batches. Call profiler.step() after each batch.
    """
    Path(config.profile_output).mkdir(parents=True, exist_ok=True)

    schedule = torch.profiler.schedule(
        wait=config.batch_profile_wait_batches,
        warmup=config.batch_profile_warmup_batches,
        active=config.batch_profile_active_batches,
        repeat=config.batch_profile_repeat_batches,
    )

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    logger.info(f"Starting BATCH profiler with activities: {activities}")
    logger.info(f"Output: {config.profile_output}")

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(config.profile_output),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        yield prof


@contextmanager
def _full_context(
    config: ProfilingConfig,
) -> Iterator[torch.profiler.profile]:
    """
    Context manager for torch.profiler WITHOUT a schedule.

    Profiles EVERYTHING within the context - no waiting, no warmup,
    just continuous profiling. Use this to capture data loading, model
    setup, compilation, and training all in one trace.

    Warning: Generates very large traces. Only use for short runs.
    """
    Path(config.profile_output).mkdir(parents=True, exist_ok=True)

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    logger.info(f"Starting FULL profiler with activities: {activities}")
    logger.info(f"Output: {config.profile_output}")

    with torch.profiler.profile(
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(config.profile_output),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        yield prof


@contextmanager
def _disabled_context() -> Iterator[None]:
    """Context manager that yields None (no profiling)."""
    yield None


def get_profiler_context(
    config: ProfilingConfig,
) -> AbstractContextManager[torch.profiler.profile | None]:
    """
    Get the appropriate profiler context based on the configuration.

    Returns a context manager that yields:
        - torch.profiler.profile for "full" or "batch" modes
        - None for "disabled" mode
    """
    if config.profile_mode == "full":
        return _full_context(config)
    if config.profile_mode == "batch":
        return _batch_context(config)
    return _disabled_context()
