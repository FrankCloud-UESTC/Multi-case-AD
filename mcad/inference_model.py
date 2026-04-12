"""Inference-only model for export (Encoder + Memory heatmap fused)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mcad.config import ModelConfig, MemoryConfig
from mcad.encoder import Encoder


class InferenceModel(nn.Module):
    """Fused inference model: Encoder + Memory heatmap computation.

    During inference, only the Encoder and Memory's heatmap logic are used.
    This module fuses them into a single forward pass with memory keys baked
    in as a buffer, making it suitable for ONNX export and torch.compile.

    Args:
        model_config: Model architecture configuration.
        memory_config: Memory module configuration.
        keys: Pre-trained memory keys tensor (M, D).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        memory_config: MemoryConfig,
        keys: torch.Tensor,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels=model_config.in_channels)
        self.memory_config = memory_config

        # Register keys as a buffer (part of model state, not a parameter)
        self.register_buffer("keys", keys)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly heatmap.

        Args:
            x: Input images (B, C, H, W).

        Returns:
            Anomaly heatmap (B, H', W', 1).
        """
        fea, _, _, _ = self.encoder(x)

        # Normalize features (same as Memory.forward test path)
        query = F.normalize(fea, dim=1)
        query = query.permute(0, 2, 3, 1)  # BCHW -> BHWC

        batch_size, h, w, dims = query.size()
        query_reshape = query.contiguous().view(batch_size * h * w, dims)

        # Compute distance to all memory keys
        query_key_dist = ((query_reshape.unsqueeze(1) - self.keys.unsqueeze(0)) ** 2).mean(dim=2)
        _, nearest_idx = torch.topk(query_key_dist, 1, dim=1, largest=False)

        # Anomaly score: sum of (query - nearest_key)^4
        heatmap = torch.sum(
            torch.pow(query_reshape - self.keys[nearest_idx.squeeze(1), :], 4),
            dim=1,
        )
        return heatmap.view(batch_size, h, w, 1)


def build_inference_model(
    checkpoint_path: str,
    keys_path: str,
    device: torch.device,
) -> InferenceModel:
    """Build an InferenceModel from a training checkpoint.

    Args:
        checkpoint_path: Path to the training checkpoint.
        keys_path: Path to the saved memory keys (.pt file).
        device: Target device.

    Returns:
        InferenceModel ready for inference or export.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config
    saved_config = None
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        saved_config = checkpoint["config"]

    model_config = saved_config.model if saved_config else ModelConfig()
    memory_config = saved_config.memory if saved_config else MemoryConfig()

    # Load keys
    keys = torch.load(keys_path, map_location=device, weights_only=False)
    if keys.dim() != 2:
        raise ValueError(f"Expected keys to be 2D tensor, got shape {keys.shape}")

    # Build inference model
    model = InferenceModel(model_config, memory_config, keys)

    # Load encoder weights from checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        # Legacy full model
        state_dict = checkpoint.state_dict()

    # Filter only encoder keys
    encoder_state = {
        k.replace("encoder.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    model.encoder.load_state_dict(encoder_state)
    model.to(device).eval()

    return model
