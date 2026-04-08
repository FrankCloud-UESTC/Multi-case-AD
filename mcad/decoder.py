"""Decoder module for the convolutional autoencoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution block: Conv-BN-ReLU x2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GenBlock(nn.Module):
    """Generation block with Tanh activation for final output."""

    def __init__(self, in_channels: int, out_channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    """Upsampling via transposed convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Decoder(nn.Module):
    """U-Net style decoder with skip connections.

    Takes the bottleneck feature and skip connections from the encoder.
    """

    def __init__(self, out_channels: int = 3) -> None:
        super().__init__()
        self.conv = ConvBlock(512, 512)
        self.up4 = UpsampleBlock(512, 256)
        self.deconv3 = ConvBlock(512, 256)
        self.up3 = UpsampleBlock(256, 128)
        self.deconv2 = ConvBlock(256, 128)
        self.up2 = UpsampleBlock(128, 64)
        self.deconv1 = GenBlock(128, out_channels, hidden=64)

    def forward(
        self,
        x: torch.Tensor,
        skip1: torch.Tensor,
        skip2: torch.Tensor,
        skip3: torch.Tensor,
    ) -> torch.Tensor:
        """Decode with skip connections.

        Args:
            x: Bottleneck feature (B, 512, H/8, W/8).
            skip1: Encoder conv1 output (B, 64, H, W).
            skip2: Encoder conv2 output (B, 128, H/2, W/2).
            skip3: Encoder conv3 output (B, 256, H/4, W/4).
        """
        x = self.conv(x)

        x = self.up4(x)
        x = self.deconv3(torch.cat([skip3, x], dim=1))

        x = self.up3(x)
        x = self.deconv2(torch.cat([skip2, x], dim=1))

        x = self.up2(x)
        output = self.deconv1(torch.cat([skip1, x], dim=1))
        return output
