"""Encoder module for the convolutional autoencoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution block: Conv-BN-ReLU x2."""

    def __init__(self, in_channels: int, out_channels: int, final_relu: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        ]
        if final_relu:
            layers.extend([
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
        else:
            layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """U-Net style encoder producing multi-scale features.

    Input:  (B, C, H, W) -> Output: (B, 512, H/8, W/8) + skip features.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = ConvBlock(256, 512, final_relu=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input and return bottleneck + skip features.

        Returns:
            (bottleneck, skip1, skip2, skip3) where skip features come from
            conv1, conv2, conv3 respectively.
        """
        s1 = self.conv1(x)
        p1 = self.pool1(s1)

        s2 = self.conv2(p1)
        p2 = self.pool2(s2)

        s3 = self.conv3(p2)
        p3 = self.pool3(s3)

        bottleneck = self.conv4(p3)
        return bottleneck, s1, s2, s3
