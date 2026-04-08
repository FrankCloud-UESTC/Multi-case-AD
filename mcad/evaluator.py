"""Evaluation pipeline for anomaly detection."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mcad.config import EvalConfig
from mcad.conv_ae import ConvAE
from mcad.data import AnomalyDataset, build_dataloader
from mcad.utils import resolve_device

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for the Memory-Augmented Convolutional Autoencoder.

    Loads a trained model and memory keys, runs inference, and saves
    anomaly heatmap visualizations.

    Args:
        config: Evaluation configuration.
    """

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)

        # Data
        dataset = AnomalyDataset(
            video_folder=config.dataset_path,
            resize_height=config.resize_height,
            resize_width=config.resize_width,
            class_name=config.class_name,
        )
        self.dataloader = build_dataloader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        )

        # Model - try to load config from checkpoint first
        checkpoint = torch.load(config.model_dir, map_location=self.device, weights_only=False)
        saved_config = None
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            saved_config = checkpoint["config"]

        if saved_config is not None:
            self.model = ConvAE(
                model_config=saved_config.model,
                memory_config=saved_config.memory,
            ).to(self.device)
        else:
            self.model = ConvAE().to(self.device)

        # Support both full model save and state_dict save
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and not any(k.startswith("module") for k in checkpoint):
            # Might be a raw state_dict
            try:
                self.model.load_state_dict(checkpoint)
            except RuntimeError:
                # Legacy: full model saved via torch.save(model, ...)
                self.model = checkpoint.to(self.device)
        else:
            # Legacy: full model saved via torch.save(model, ...)
            self.model = checkpoint.to(self.device)

        # Memory keys
        m_items = torch.load(config.m_items_dir, map_location=self.device, weights_only=False)
        self.m_items = [m_items] if m_items.dim() == 2 else m_items

        self.model.eval()
        logger.info("Loaded model from %s, %d test samples", config.model_dir, len(dataset))

    def evaluate(self) -> None:
        """Run evaluation and save heatmap visualizations."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Evaluating %d samples", len(self.dataloader))

        with torch.no_grad():
            for imgs, label, name in tqdm(self.dataloader, desc="Evaluating"):
                imgs = imgs.to(self.device)

                heatmap, fea, _, _ = self.model.forward(
                    imgs, self.m_items, train=False, has_defect=False,
                )

                # Convert image for visualization
                img_np = imgs[0].cpu().detach().numpy()
                img_np = np.moveaxis(img_np, 0, 2)  # CHW -> HWC
                img_np = (img_np + 1) * 127 if img_np.min() < 0 else img_np * 255

                # Resize heatmap to original image dimensions
                mask = heatmap[0, :, :, 0].detach().cpu().numpy()
                mask = cv2.resize(mask, (self.config.resize_width, self.config.resize_height))

                # Save visualization
                img_name = str(name[0])
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                ax.imshow(img_np[:, :, 0] if img_np.ndim == 3 else img_np, cmap="gray")
                ax.imshow(mask, cmap="jet", alpha=0.2)
                fig.savefig(output_dir / img_name, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

        logger.info("Results saved to %s", output_dir)
