"""
Profiling utilities for PyTorch training.

Usage with train.py:
    # Profile training batches with torch.profiler (step-based schedule)
    uv run torchrun --nproc_per_node=1 dbtransformer/bin/train.py \
        --profile torch --no-wandb --epochs 1

    # Profile EVERYTHING (data loading, setup, training, etc.)
    uv run torchrun --nproc_per_node=1 dbtransformer/bin/train.py \
        --profile torch-full --no-wandb --epochs 1

    # Profile with MPS profiler (for Metal-level GPU analysis)
    uv run torchrun --nproc_per_node=1 dbtransformer/bin/train.py \
        --profile mps --no-wandb --epochs 1

    # Profile with cProfile (Python-level only)
    uv run python -m cProfile -o profile.prof -s cumtime \
        dbtransformer/bin/train.py --no-wandb --epochs 1

After running with --profile torch or --profile torch-full:
    tensorboard --logdir=./profiler_logs

After running with --profile mps:
    Open Instruments.app -> Logging template to view signpost traces
"""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import torch
from loguru import logger


@contextmanager
def torch_profiler_context(
    output_dir: str = "./profiler_logs",
    wait: int = 2,
    warmup: int = 2,
    active: int = 6,
    repeat: int = 1,
) -> Generator[torch.profiler.profile | None]:
    """
    Context manager for torch.profiler with TensorBoard output.

    Args:
        output_dir: Directory to save profiler traces
        wait: Steps to skip before warmup
        warmup: Steps for warmup (not recorded)
        active: Steps to actively record
        repeat: Number of cycles to repeat
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    schedule = torch.profiler.schedule(
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
    )

    # Determine activities based on available hardware
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    logger.info(f"Starting torch.profiler with activities: {activities}")
    logger.info(f"Schedule: wait={wait}, warmup={warmup}, active={active}, repeat={repeat}")
    logger.info(f"Output: {output_dir}")

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        yield prof

    logger.success(f"Profiling complete. View with: tensorboard --logdir={output_dir}")


@contextmanager
def mps_profiler_context(
    mode: str = "interval",
    wait_until_completed: bool = False,
) -> Generator[None]:
    """
    Context manager for MPS profiler (Apple Metal).

    Args:
        mode: "interval", "event", or "interval,event"
        wait_until_completed: Whether to wait for GPU ops to complete
    """
    if not torch.backends.mps.is_available():
        logger.warning("MPS not available, skipping MPS profiler")
        yield
        return

    import torch.mps.profiler as mps_profiler  # noqa: PLC0415

    logger.info(f"Starting MPS profiler (mode={mode})")
    logger.info("After training, open Instruments.app -> Logging template")

    mps_profiler.start(mode=mode, wait_until_completed=wait_until_completed)
    try:
        yield
    finally:
        mps_profiler.stop()
        logger.success("MPS profiling complete. Open Instruments.app -> Logging template")


@contextmanager
def torch_profiler_full_context(
    output_dir: str = "./profiler_logs",
) -> Generator[torch.profiler.profile | None]:
    """
    Context manager for torch.profiler WITHOUT a schedule.

    This profiles EVERYTHING within the context - no waiting, no warmup,
    just continuous profiling. Use this to capture data loading, model
    setup, compilation, and training all in one trace.

    Args:
        output_dir: Directory to save profiler traces
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine activities based on available hardware
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    logger.info(f"Starting FULL torch.profiler with activities: {activities}")
    logger.info("Profiling everything (no schedule - always active)")
    logger.info(f"Output: {output_dir}")

    with torch.profiler.profile(
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        yield prof

    logger.success(f"Profiling complete. View with: tensorboard --logdir={output_dir}")


@contextmanager
def no_profiler_context() -> Generator[None]:
    """Dummy context manager for no profiling."""
    yield None
