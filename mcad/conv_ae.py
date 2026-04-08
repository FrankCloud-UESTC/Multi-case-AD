"""Convolutional Autoencoder with Memory module for anomaly detection."""

from __future__ import annotations

import torch
import torch.nn as nn

from mcad.config import ModelConfig, MemoryConfig
from mcad.encoder import Encoder
from mcad.decoder import Decoder
from mcad.memory import Memory


class ConvAE(nn.Module):
    """Memory-augmented Convolutional Autoencoder.

    Combines Encoder + Decoder + Memory module. During training, skip connections
    are zeroed to force the model to reconstruct from memory-read features.

    Args:
        model_config: Model architecture configuration.
        memory_config: Memory module configuration.
    """

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        memory_config: MemoryConfig | None = None,
    ) -> None:
        super().__init__()
        model_config = model_config or ModelConfig()
        memory_config = memory_config or MemoryConfig()

        self.encoder = Encoder(in_channels=model_config.in_channels)
        self.decoder = Decoder(out_channels=model_config.in_channels)
        self.memory = Memory(memory_config)

        # Pixel-level classification head (optional, not used in current loss)
        self.cls_head = nn.Conv2d(model_config.feature_dim, 1, kernel_size=1)
        self.cls_activation = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        keys: torch.Tensor,
        train: bool = True,
        has_defect: bool = False,
        label: list[torch.Tensor] | None = None,
        epoch: int = 0,
        defect_memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass.

        Args:
            x: Input images (B, C, H, W).
            keys: Memory bank tensor (M, D).
            train: Training mode flag.
            has_defect: Whether sample contains defects.
            label: Label tensors for defect handling.
            epoch: Current training epoch.
            defect_memory: Defect feature bank (N, D).

        Returns:
            Training: (output, pred, features, separateness_loss, compactness_loss, keys, defect_memory)
            Testing: (heatmap, features, keys, defect_memory)
        """
        fea, skip1, skip2, skip3 = self.encoder(x)
        pred = self.cls_activation(self.cls_head(fea))

        if train:
            updated_fea, separate_loss, compact_loss, keys, defect_memory = self.memory(
                [fea], keys, train=True, has_defect=has_defect,
                label=label, epoch=epoch, defect_memory=defect_memory,
            )
            # Zero out skip connections during training
            output = self.decoder(
                updated_fea,
                torch.zeros_like(skip1),
                torch.zeros_like(skip2),
                torch.zeros_like(skip3),
            )
            return output, pred, fea, separate_loss, compact_loss, keys, defect_memory

        # Test mode
        heatmap, _, _, keys, defect_memory = self.memory(
            [fea], keys, train=False, has_defect=has_defect,
            label=label, epoch=epoch, defect_memory=defect_memory,
        )
        return heatmap, fea, keys, defect_memory
