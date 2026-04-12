"""Utility functions: device resolution, seed management, logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch


def resolve_device(device_spec: str) -> torch.device:
    """Resolve a device specification to a torch.device.

    Args:
        device_spec: One of "auto", "cpu", or "cuda:N".

    Returns:
        Resolved torch.device.
    """
    if device_spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    return torch.device(device_spec)


def get_amp_device_type(device: torch.device) -> str:
    """Get the appropriate device type string for torch.amp.autocast.

    Returns "cuda" for CUDA devices, "cpu" otherwise.
    """
    return "cuda" if device.type == "cuda" else "cpu"


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all relevant libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Ensure deterministic CUDA operations (may reduce performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    """Configure logging for the entire project.

    Args:
        level: Logging level (default: INFO).
        log_file: Optional file path to also log to.
    """
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_file), encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    # Reduce noise from third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
