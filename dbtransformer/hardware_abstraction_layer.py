"""
Device configuration abstraction for cross-platform training.

Centralizes all device-specific decisions (CUDA vs MPS vs CPU) in one place,
eliminating scattered conditionals throughout the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from loguru import logger
from transformers.utils.import_utils import is_flash_attn_2_available


@dataclass(frozen=True)
class HardwareConfig:
    """
    Immutable configuration for device-specific behavior.

    All device-dependent decisions are made once at initialization,
    then this config is passed through to model/training code.

    We anticipate running locally on Apple Silicon (MPS), and then "for real" on CUDA.
    """

    # The accelerator device to use.
    device: torch.device

    # For our relational transformer, the type for float params.
    model_dtype: torch.dtype

    # Whether to use flex attention masks or not - unsupported on MPS/CPU.
    use_flex_attention: bool

    # Whether to use fused optimizer or not.
    use_fused_optimizer: bool

    # Whether to pin memory or not.
    pin_memory: bool

    # The DDP backend to use - "gloo" (CPU, MPS) or "nccl" (CUDA)
    ddp_backend: Literal["gloo", "nccl"]

    # Whether to use flash attention on the embedder model.
    use_flash_attention: bool

    # The float dtype for the embedder model.
    embedder_dtype: torch.dtype

    @classmethod
    def auto_detect(cls, local_rank: int = 0) -> HardwareConfig:
        """
        Auto-detect the best device configuration for the current hardware.

        Args:
            local_rank: GPU index for multi-GPU setups (ignored on MPS/CPU).

        Returns:
            DeviceConfig optimized for the detected hardware.
        """
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            return cls(
                device=torch.device(f"cuda:{local_rank}"),
                model_dtype=torch.bfloat16,
                use_flex_attention=True,
                use_fused_optimizer=True,
                pin_memory=True,
                ddp_backend="nccl",
                use_flash_attention=is_flash_attn_2_available(),
                embedder_dtype=torch.float16,
            )

        if torch.backends.mps.is_available():
            return cls(
                device=torch.device("mps"),
                model_dtype=torch.float32,
                use_flex_attention=False,
                use_fused_optimizer=False,
                pin_memory=False,
                ddp_backend="gloo",
                use_flash_attention=False,
                embedder_dtype=torch.float16,
            )

        # CPU fallback
        return cls(
            device=torch.device("cpu"),
            model_dtype=torch.float32,
            use_flex_attention=False,
            use_fused_optimizer=False,
            pin_memory=False,
            ddp_backend="gloo",
            use_flash_attention=False,
            embedder_dtype=torch.float16,
        )

    @classmethod
    def for_cuda(cls, local_rank: int = 0) -> HardwareConfig:
        """Force CUDA configuration (fails if CUDA unavailable)."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        torch.cuda.set_device(local_rank)
        return cls(
            device=torch.device(f"cuda:{local_rank}"),
            model_dtype=torch.bfloat16,
            use_flex_attention=True,
            use_fused_optimizer=True,
            pin_memory=True,
            ddp_backend="nccl",
            use_flash_attention=is_flash_attn_2_available(),
            embedder_dtype=torch.float16,
        )

    @classmethod
    def for_mps(cls) -> HardwareConfig:
        """Force MPS configuration (fails if MPS unavailable)."""
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return cls(
            device=torch.device("mps"),
            model_dtype=torch.float32,
            use_flex_attention=False,
            use_fused_optimizer=False,
            pin_memory=False,
            ddp_backend="gloo",
            use_flash_attention=False,
            embedder_dtype=torch.float16,
        )

    @classmethod
    def for_cpu(cls) -> HardwareConfig:
        return cls(
            device=torch.device("cpu"),
            model_dtype=torch.float32,
            use_flex_attention=False,
            use_fused_optimizer=False,
            pin_memory=False,
            ddp_backend="gloo",
            use_flash_attention=False,
            embedder_dtype=torch.float16,
        )

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA."""
        return self.device.type == "cuda"

    @property
    def is_mps(self) -> bool:
        """Check if using MPS (Apple Silicon)."""
        return self.device.type == "mps"

    @property
    def is_cpu(self) -> bool:
        """Check if using CPU."""
        return self.device.type == "cpu"

    @property
    def local_rank(self) -> int:
        """Get local rank (GPU index) for CUDA, 0 otherwise."""
        if self.is_cuda:
            return self.device.index or 0
        return 0

    def log_config(self) -> None:
        """Log the device configuration."""
        logger.info(
            f"HardwareConfig: device={self.device}, model_dtype={self.model_dtype}, "
            f"using_flex_attn={self.use_flex_attention}, "
            f"fused_optim={self.use_fused_optimizer}, "
            f"backend={self.ddp_backend}, "
            f"use_flash_attention={self.use_flash_attention}, "
            f"embedder_dtype={self.embedder_dtype}"
        )
