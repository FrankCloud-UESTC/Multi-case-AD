"""CUDA kernel examples and utilities for the mcad anomaly detection project.

This module provides custom CUDA kernels that can be JIT-compiled and used
as drop-in replacements for PyTorch operations in the inference pipeline.

Before using, call setup_cuda_build_env() from mcad.cuda_utils.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_CUDA_KERNELS_DIR = str(Path(__path__[0]) / ".." / "cuda_kernels")

# Cache compiled modules
_vec_add_module = None
_dist_heatmap_module = None


def vec_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vector addition using a custom CUDA kernel (demo).

    Args:
        a: First input tensor (must be on CUDA).
        b: Second input tensor (same shape as a, must be on CUDA).

    Returns:
        Element-wise a + b computed via custom kernel.
    """
    global _vec_add_module
    if _vec_add_module is None:
        _vec_add_module = load(
            name="vec_add_cuda",
            sources=[str(Path(_CUDA_KERNELS_DIR) / "vec_add_cuda.cu")],
            extra_cuda_cflags=["-allow-unsupported-compiler"],
        )
    return _vec_add_module.vec_add(a, b)


def _get_dist_heatmap_module():
    global _dist_heatmap_module
    if _dist_heatmap_module is None:
        _dist_heatmap_module = load(
            name="dist_heatmap_cuda",
            sources=[str(Path(_CUDA_KERNELS_DIR) / "dist_heatmap_cuda.cu")],
            extra_cuda_cflags=["-allow-unsupported-compiler"],
        )
    return _dist_heatmap_module


def compute_heatmap_fused(
    query: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    """Fused distance + top1 + power-sum kernel for anomaly heatmap (naive).

    Replaces the multi-step PyTorch implementation with a single fused CUDA
    kernel that:
    1. Computes L2 distance from each query to all keys
    2. Finds the nearest key
    3. Computes sum((query - nearest_key)^4) per pixel

    Args:
        query: Feature map (B, H, W, D) after normalization and permute.
        keys: Memory keys (M, D).

    Returns:
        Anomaly heatmap (B, H, W, 1).
    """
    mod = _get_dist_heatmap_module()
    return mod.compute_heatmap(query, keys)


def compute_heatmap_fused_tiled(
    query: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    """Fused distance + top1 + power-sum kernel for anomaly heatmap (tiled).

    Same functionality as compute_heatmap_fused but uses shared memory tiling
    for keys, reducing global memory traffic by a factor of blockDim.x.
    Better performance for larger N (spatial dimensions).

    Args:
        query: Feature map (B, H, W, D) after normalization and permute.
        keys: Memory keys (M, D).

    Returns:
        Anomaly heatmap (B, H, W, 1).
    """
    mod = _get_dist_heatmap_module()
    return mod.compute_heatmap_tiled(query, keys)
