"""CUDA kernel development utilities for the mcad project."""

from __future__ import annotations

import os
from pathlib import Path

# MSVC and CUDA paths - adjust if your installation differs
MSVC_BASE = r"D:\Downloads\vs_cpp\VC\Tools\MSVC\14.50.35717"
MSVC_BIN = os.path.join(MSVC_BASE, "bin", "Hostx64", "x64")
MSVC_INC = os.path.join(MSVC_BASE, "include")
MSVC_LIB = os.path.join(MSVC_BASE, "lib", "x64")


def setup_cuda_build_env() -> None:
    """Set up environment variables needed for CUDA kernel compilation on Windows.

    Call this before using torch.utils.cpp_extension.load() or building
    custom CUDA kernels. Required because:
    - MSVC cl.exe may not be in PATH
    - CUDA 12.4 doesn't recognize MSVC 14.50+ by default

    Usage:
        from mcad.cuda_utils import setup_cuda_build_env
        setup_cuda_build_env()
        from torch.utils.cpp_extension import load
        module = load(...)
    """
    # Add MSVC and conda Scripts to PATH
    paths_to_add = [
        MSVC_BIN,
        r"D:\miniconda3\envs\mcad\Scripts",
    ]
    for p in paths_to_add:
        if p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

    # MSVC include/lib
    os.environ["INCLUDE"] = MSVC_INC
    os.environ["LIB"] = MSVC_LIB

    # Fix: MSVC 14.50+ is not recognized by CUDA 12.4
    # Pass -allow-unsupported-compiler via extra_cuda_cflags when calling load()
