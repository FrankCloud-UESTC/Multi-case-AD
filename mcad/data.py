"""Data pipeline: Dataset, transforms, and DataLoader factory."""

from __future__ import annotations

import glob
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader

logger = logging.getLogger(__name__)


def load_frame(
    filename: str,
    resize_height: int,
    resize_width: int,
    class_name: str,
) -> tuple[np.ndarray, list[torch.Tensor], str]:
    """Load a single image and its corresponding label.

    Args:
        filename: Path to the image file.
        resize_height: Target height for resizing.
        resize_width: Target width for resizing.
        class_name: Substring used to derive the label directory path.

    Returns:
        Tuple of (image_array, label_tensors, image_name).
        Image array is in CHW format, normalized to [0, 1].
    """
    image = cv2.imread(filename)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {filename}")

    # Derive label path
    label_dir = filename.replace(class_name, "labels").rsplit(".", 1)[0] + ".png"
    if os.path.exists(label_dir):
        label = cv2.imread(label_dir)
        label = label[:, :, :1]
        label = label / np.max(label)
    else:
        label = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)

    # Resize and normalize image
    image_resized = cv2.resize(image, (resize_width, resize_height))
    image_resized = image_resized.astype(np.float32)
    image_resized = (image_resized - image_resized.min()) / (
        image_resized.max() - image_resized.min() + 1e-8
    )
    image_resized = np.moveaxis(image_resized, 2, 0)  # HWC -> CHW

    # Multi-scale labels (1/8 and 1/4 of original)
    labels: list[torch.Tensor] = []
    for divisor in [8, 4]:
        label_ = cv2.resize(label, (resize_width // divisor, resize_height // divisor))
        label_ = label_.astype(np.float32)
        if label_.ndim == 2:
            label_ = np.expand_dims(label_, 2)
        label_ = np.moveaxis(label_, 2, 0)  # HWC -> CHW
        label_ = (label_ > 0).astype(np.float32)
        labels.append(torch.from_numpy(label_))

    img_name = os.path.basename(filename)
    return image_resized, labels, img_name


class AnomalyDataset(TorchDataset):
    """Dataset for multi-case anomaly detection.

    Scans a directory of subfolders (each subfolder = one case),
    collects all PNG frames, and returns (image, labels, filename) tuples.
    """

    def __init__(
        self,
        video_folder: str,
        resize_height: int = 128,
        resize_width: int = 128,
        class_name: str = "",
    ) -> None:
        self.dir = video_folder
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.class_name = class_name
        self.videos: OrderedDict[str, dict] = OrderedDict()
        self._setup()
        self.samples = self._get_all_samples()
        logger.info("AnomalyDataset: %d samples from %s", len(self.samples), video_folder)

    def _setup(self) -> None:
        videos = sorted(glob.glob(os.path.join(self.dir, "*")))
        for video in videos:
            video_name = os.path.basename(video)
            self.videos[video_name] = {
                "path": video,
                "frame": sorted(glob.glob(os.path.join(video, "*.png"))),
            }

    def _get_all_samples(self) -> list[str]:
        frames: list[str] = []
        for video_name in self.videos:
            frames.extend(self.videos[video_name]["frame"])
        return frames

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[torch.Tensor], str]:
        filepath = self.samples[index]
        image, labels, img_name = load_frame(
            filepath,
            self.resize_height,
            self.resize_width,
            self.class_name,
        )
        return torch.from_numpy(image), labels, img_name

    def __len__(self) -> int:
        return len(self.samples)


def build_dataloader(
    dataset: AnomalyDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
) -> TorchDataLoader:
    """Build a PyTorch DataLoader from an AnomalyDataset."""
    return TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
