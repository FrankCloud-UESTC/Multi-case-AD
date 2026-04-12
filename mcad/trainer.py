"""Training loop with proper checkpointing, logging, and AMP support."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from tqdm import tqdm

from mcad.config import TrainConfig
from mcad.conv_ae import ConvAE
from mcad.data import AnomalyDataset, build_dataloader
from mcad.utils import resolve_device, set_seed, get_amp_device_type

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for the Memory-Augmented Convolutional Autoencoder.

    Handles the training loop with:
    - Mixed precision (AMP) support
    - Gradient clipping
    - Checkpoint management (save best / save last / resume)
    - Structured logging

    Args:
        config: Training configuration.
    """

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        set_seed(config.seed)

        # Data
        dataset = AnomalyDataset(
            video_folder=config.data.dataset_path,
            resize_height=config.data.resize_height,
            resize_width=config.data.resize_width,
            class_name=config.data.class_name,
        )
        self.dataloader = build_dataloader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True,
        )

        # Model
        self.model = ConvAE(
            model_config=config.model,
            memory_config=config.memory,
        ).to(self.device)

        # Optimizer: only encoder + decoder parameters
        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        self.optimizer = optim.Adam(
            params,
            lr=config.optim.learning_rate,
            weight_decay=config.optim.weight_decay,
        )

        # Scheduler
        self.scheduler = self._build_scheduler()

        # Loss
        self.mse_loss = nn.MSELoss(reduction="none")

        # AMP
        self.amp_device_type = get_amp_device_type(self.device)
        self.scaler = torch.amp.GradScaler(self.amp_device_type, enabled=config.amp)

        # Output directory
        self.exp_dir = Path(config.exp_dir) / config.data.dataset_type / "log"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config.to_yaml(self.exp_dir / "config.yaml")

        # State
        self.m_items: torch.Tensor | None = None
        self.defect_memory: torch.Tensor | None = None
        self.current_epoch: int = 0
        self.best_loss: float = float("inf")

    def _build_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        cfg = self.config.optim
        if cfg.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.t_max)
        elif cfg.scheduler == "step":
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        else:
            return optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0)

    def _init_memory(self) -> torch.Tensor:
        """Initialize memory items as normalized random vectors."""
        m_items = F.normalize(
            torch.rand(
                (self.config.model.memory_size, self.config.model.key_dim),
                dtype=torch.float,
            ),
            dim=1,
        )
        return m_items.to(self.device)

    @staticmethod
    def downsample_memory(m_items: torch.Tensor, n_clusters: int = 10) -> torch.Tensor:
        """Compress memory bank via KMeans clustering."""
        device = m_items.device
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(
            m_items.detach().cpu().numpy()
        )
        return torch.tensor(kmeans.cluster_centers_, device=device, dtype=m_items.dtype)

    def train(self) -> None:
        """Run the full training loop."""
        logger.info(
            "Training on %s, %d samples, device=%s",
            self.config.data.dataset_type,
            len(self.dataloader) * self.config.batch_size,
            self.device,
        )

        self.m_items = self._init_memory()
        self.defect_memory = torch.empty(
            0, self.config.model.key_dim, device=self.device
        )

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            self.model.train()

            epoch_stats = self._train_epoch(epoch)

            # Reset defect memory each epoch
            self.defect_memory = torch.empty(
                0, self.config.model.key_dim, device=self.device
            )

            # Log epoch summary
            logger.info(
                "Epoch %d/%d | Recon: %.6f | Compact: %.6f | Separate: %.6f | Total: %.6f",
                epoch + 1,
                self.config.epochs,
                epoch_stats["recon"],
                epoch_stats["compact"],
                epoch_stats["separate"],
                epoch_stats["total"],
            )

            # Checkpoint
            is_best = epoch_stats["total"] < self.best_loss
            if is_best:
                self.best_loss = epoch_stats["total"]
            self._save_checkpoint(epoch, is_best)

            self.scheduler.step()

        logger.info("Training finished. Best loss: %.6f", self.best_loss)

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch and return statistics."""
        loss_pixels = []
        loss_comps = []
        loss_separs = []
        loss_totals = []

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
        for batch_idx, (imgs, labels, name) in enumerate(pbar):
            imgs = imgs.to(self.device)
            labels = [l.to(self.device) for l in labels]

            has_defect = labels[0].sum().item() > 0

            with torch.amp.autocast(self.amp_device_type, enabled=self.config.amp):
                outputs, pred, fea, separate_loss, compact_loss, self.m_items, self.defect_memory = (
                    self.model.forward(
                        imgs,
                        self.m_items,
                        train=True,
                        has_defect=has_defect,
                        label=labels,
                        epoch=epoch,
                        defect_memory=self.defect_memory,
                    )
                )
                recon_loss = torch.mean(self.mse_loss(outputs, imgs))
                total_loss = (
                    self.config.loss_compact * compact_loss
                    + self.config.loss_separate * separate_loss
                    + recon_loss
                )

            # Backward with gradient clipping
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            if self.config.optim.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
                    self.config.optim.grad_clip_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Track losses
            loss_pixels.append(recon_loss.item())
            if compact_loss.item() != 0:
                loss_comps.append(compact_loss.item())
            if separate_loss.item() != 0:
                loss_separs.append(separate_loss.item())
            loss_totals.append(total_loss.item())

            # Update progress bar
            pbar.set_postfix(recon=f"{recon_loss.item():.4f}")

            if (batch_idx + 1) % self.config.log_interval == 0:
                logger.debug(
                    "Epoch %d Batch %d | Recon: %.6f | Compact: %.6f | Separate: %.6f",
                    epoch + 1, batch_idx + 1,
                    recon_loss.item(),
                    compact_loss.item(),
                    separate_loss.item(),
                )

        return {
            "recon": float(np.mean(loss_pixels)) if loss_pixels else 0.0,
            "compact": float(np.mean(loss_comps)) if loss_comps else 0.0,
            "separate": float(np.mean(loss_separs)) if loss_separs else 0.0,
            "total": float(np.mean(loss_totals)) if loss_totals else 0.0,
        }

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save training checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "m_items": self.m_items,
            "best_loss": self.best_loss,
            "config": self.config,
        }

        # Save last checkpoint
        last_path = self.exp_dir / "checkpoint_last.pth"
        torch.save(state, last_path)

        # Save best checkpoint
        if is_best:
            best_path = self.exp_dir / "checkpoint_best.pth"
            torch.save(state, best_path)
            logger.debug("Saved best checkpoint (loss=%.6f)", self.best_loss)

        # Always save memory keys separately for easy access
        torch.save(self.m_items, self.exp_dir / "keys.pt")

    def resume(self, checkpoint_path: str | Path) -> None:
        """Resume training from a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.m_items = checkpoint["m_items"].to(self.device)
        self.current_epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint["best_loss"]
        logger.info("Resumed from epoch %d, best loss %.6f", self.current_epoch, self.best_loss)
