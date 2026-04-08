"""Memory module with dynamic memory bank for anomaly detection."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from mcad.config import MemoryConfig

logger = logging.getLogger(__name__)


class Memory(nn.Module):
    """Memory-augmented module with dynamic memory bank.

    Maintains a growing memory bank of normal (background) feature prototypes.
    During training, features are clustered via KMeans and the memory bank is
    updated with pruning of items too close to known defect features.

    Args:
        config: MemoryConfig with all hyperparameters.
    """

    def __init__(self, config: MemoryConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        query: list[torch.Tensor],
        keys: torch.Tensor | list[torch.Tensor],
        train: bool = True,
        has_defect: bool = False,
        label: list[torch.Tensor] | None = None,
        epoch: int = 0,
        defect_memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through memory.

        Args:
            query: List of feature maps, each (B, C, H, W).
            keys: Memory bank tensor (M, D) or list containing one tensor (for eval).
            train: Whether in training mode.
            has_defect: Whether current sample contains defect pixels.
            label: List of label tensors (only used when has_defect=True).
            epoch: Current epoch number.
            defect_memory: Tensor of defect feature vectors (N_defect, D).

        Returns:
            If train: (updated_query, separateness_loss, compactness_loss, updated_keys, updated_defect_memory)
            If test: (heatmap, zero_loss, zero_loss, keys, defect_memory)
        """
        query = [F.normalize(q, dim=1) for q in query]
        query = [q.permute(0, 2, 3, 1) for q in query]  # BCHW -> BHWC

        if train:
            separateness_loss = torch.tensor(0.0, device=query[0].device)
            compactness_loss = torch.tensor(0.0, device=query[0].device)
            if defect_memory is None:
                defect_memory = torch.empty(0, query[0].shape[-1], device=query[0].device)

            for i, q in enumerate(query):
                s_loss, c_loss, keys, defect_memory = self._gather_loss(
                    q, True, has_defect, label[i] if label else None,
                    keys, epoch, defect_memory,
                )
                separateness_loss = separateness_loss + s_loss
                compactness_loss = compactness_loss + c_loss

            updated_query = self._read(query[0])
            return updated_query, separateness_loss, compactness_loss, keys, defect_memory

        # Test mode: compute anomaly heatmaps
        hotmaps = []
        for q in query:
            hotmap = self._gather_loss(q, False, has_defect, None, keys, epoch, None)
            hotmaps.append(hotmap)
        return hotmaps[0], torch.tensor(0.0), torch.tensor(0.0), keys, defect_memory

    def _gather_loss(
        self,
        query: torch.Tensor,
        train: bool,
        has_defect: bool,
        labels: torch.Tensor | None,
        keys: torch.Tensor,
        epoch: int,
        defect_bank: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gathering/compactness losses (train) or anomaly heatmap (test)."""
        batch_size, h, w, dims = query.size()
        device = query.device
        cfg = self.config

        if not train:
            return self._compute_heatmap(query, keys)

        # --- Training ---
        loss_normal = nn.TripletMarginLoss(margin=cfg.triplet_margin_normal)
        loss_defect = nn.TripletMarginLoss(margin=cfg.triplet_margin_defect)
        loss_mse = nn.MSELoss()
        query_reshape = query.contiguous().view(batch_size * h * w, dims)

        if has_defect and labels is not None:
            return self._train_with_defect(
                query_reshape, labels, keys, epoch, defect_bank,
                loss_defect, loss_mse, device,
            )

        # --- Normal sample training ---
        return self._train_normal(
            query_reshape, keys, epoch, defect_bank,
            loss_normal, loss_mse, device,
        )

    def _train_normal(
        self,
        query_reshape: torch.Tensor,
        keys: torch.Tensor,
        epoch: int,
        defect_bank: torch.Tensor,
        loss_triplet: nn.Module,
        loss_mse: nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training logic for normal (non-defect) samples."""
        cfg = self.config
        dims = query_reshape.shape[-1]

        kmeans = KMeans(n_clusters=cfg.n_clusters, random_state=0, n_init="auto").fit(
            query_reshape.detach().cpu().numpy()
        )
        center = torch.tensor(kmeans.cluster_centers_, device=device, dtype=query_reshape.dtype)

        # Compactness: push each query toward its cluster center
        compactness = torch.tensor(0.0, device=device)
        for i in range(query_reshape.shape[0]):
            compactness = compactness + loss_mse(
                query_reshape[[i], :], center[[kmeans.labels_[i]], :]
            )

        # Memory bank update
        if epoch >= 1:
            temp_defect = torch.empty(0, dims, device=device)
            keys, defect_bank = self._memory_items_operation(keys, center, False, None, defect_bank or temp_defect)
        if keys.shape[0] == 0:
            keys = center
        if epoch < 1:
            keys = torch.cat((keys, center), dim=0)

        # Separateness: triplet loss with two nearest centers
        query_center_dist = ((query_reshape.unsqueeze(1) - center.unsqueeze(0)) ** 2).mean(dim=2)
        _, top2_indices = torch.topk(query_center_dist, 2, dim=1, largest=False)

        separateness = torch.tensor(0.0, device=device)
        for i in range(query_reshape.shape[0]):
            separateness = separateness + loss_triplet(
                query_reshape[[i], :],
                center[[top2_indices[i, 0]], :],
                center[[top2_indices[i, 1]], :],
            )

        # Additional separateness from defect memory
        if defect_bank is not None and defect_bank.shape[0] > 0:
            query_defect_dist = ((query_reshape.unsqueeze(1) - defect_bank.unsqueeze(0)) ** 2).mean(dim=2)
            _, near_defect_idx = torch.topk(query_defect_dist, 1, dim=1, largest=False)
            for i in range(query_reshape.shape[0]):
                separateness = separateness + loss_triplet(
                    query_reshape[[i], :],
                    center[[kmeans.labels_[i]], :],
                    defect_bank[[near_defect_idx[i, 0]], :],
                )

        return separateness, compactness, keys, defect_bank if defect_bank is not None else torch.empty(0, dims, device=device)

    def _train_with_defect(
        self,
        query_reshape: torch.Tensor,
        labels: torch.Tensor,
        keys: torch.Tensor,
        epoch: int,
        defect_bank: torch.Tensor,
        loss_defect: nn.Module,
        loss_mse: nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training logic for defect-containing samples."""
        cfg = self.config
        dims = query_reshape.shape[-1]

        labels_2d = labels[:, 0, :, :]  # (B, H', W')
        labels_3d = labels_2d.unsqueeze(-1)  # (B, H', W', 1)
        label_reshape = labels_3d.contiguous().view(-1, 1)

        pos_index = (label_reshape == 0).squeeze()
        neg_index = (label_reshape != 0).squeeze()
        pos = query_reshape[pos_index].to(device)
        neg = query_reshape[neg_index].to(device)

        if pos.shape[0] == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), keys, defect_bank

        # Cluster background features
        bg_kmeans = KMeans(n_clusters=min(cfg.n_clusters, pos.shape[0]), random_state=0, n_init="auto").fit(
            pos.detach().cpu().numpy()
        )
        bg_centers = torch.tensor(bg_kmeans.cluster_centers_, device=device, dtype=query_reshape.dtype)

        # Memory bank initialization
        if keys.shape[0] == 0:
            keys = bg_centers
        if epoch < 1:
            keys = torch.cat((keys, bg_centers), dim=0)
        if epoch >= 1:
            keys, defect_bank = self._memory_items_operation(keys, bg_centers, True, neg, defect_bank)

        # Gathering loss: background vs. defect triplet
        gathering_loss = torch.tensor(0.0, device=device)
        if neg.shape[0] > 0:
            neg_anchor = neg.to(device)
            for j in range(pos.shape[0]):
                gathering_loss = gathering_loss + loss_defect(
                    pos[[j], :],
                    bg_centers[[bg_kmeans.labels_[j]], :],
                    neg_anchor,
                )

        # Compactness loss
        compact_loss = torch.tensor(0.0, device=device)
        for k in range(neg.shape[0]):
            central = bg_centers[[bg_kmeans.labels_[k]], :]
            compact_loss = compact_loss + loss_mse(pos[[k], :], central)

        scale = math.log(epoch + 2, 2)
        return scale * gathering_loss, compact_loss, keys, defect_bank

    def _compute_heatmap(
        self, query: torch.Tensor, keys: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute per-pixel anomaly heatmap for testing."""
        batch_size, h, w, dims = query.size()
        query_reshape = query.contiguous().view(batch_size * h * w, dims)

        # Find nearest memory key for each query
        query_key_dist = ((query_reshape.unsqueeze(1) - keys[0].unsqueeze(0)) ** 2).mean(dim=2)
        _, nearest_idx = torch.topk(query_key_dist, 1, dim=1, largest=False)

        # Anomaly score: sum of (query - nearest_key)^4
        heatmap = torch.sum(
            torch.pow(query_reshape - keys[0][nearest_idx.squeeze(), :].detach(), 4),
            dim=1,
        )
        return heatmap.view(batch_size, h, w, 1)

    def _read(self, query: torch.Tensor) -> torch.Tensor:
        """Read operation: reshape query for decoder input."""
        batch_size, h, w, dims = query.size()
        return query.permute(0, 3, 1, 2)  # BHWC -> BCHW

    def _memory_items_operation(
        self,
        keys: torch.Tensor,
        bg_centers: torch.Tensor,
        has_defect: bool,
        neg: torch.Tensor | None,
        defect_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update memory bank with new background centers, pruning near-defect items."""
        cfg = self.config
        dims = keys.shape[-1]
        device = keys.device
        mse_loss = nn.MSELoss()

        if has_defect and neg is not None:
            defect_memory = torch.cat((defect_memory, neg), dim=0)
            logger.debug("Defect bank shape after adding neg: %s", defect_memory.shape)

        if defect_memory.shape[0] > 0:
            # Remove memory keys that are too close to defect features
            keys_defect_dist = ((keys.unsqueeze(1) - defect_memory.unsqueeze(0)) ** 2).mean(dim=2)
            _, near_defect_idx = torch.topk(keys_defect_dist, 1, dim=0, largest=False)

            drop_indices = []
            for i in range(near_defect_idx.shape[1]):
                dist = torch.sqrt(mse_loss(keys[near_defect_idx[0, i], :], defect_memory[i, :]))
                if dist < cfg.defect_mse_threshold:
                    drop_indices.append(near_defect_idx[0, i].item())

            logger.debug("Dropping %d memory keys near defects", len(drop_indices))
            if drop_indices:
                keep_mask = torch.ones(keys.shape[0], dtype=torch.bool, device=device)
                keep_mask[drop_indices] = False
                keys = keys[keep_mask]

            # Filter out bg_centers too close to defect memory
            drop_center_indices = set()
            for i in range(bg_centers.shape[0]):
                for j in range(defect_memory.shape[0]):
                    dist = torch.sqrt(mse_loss(bg_centers[[i], :], defect_memory[[j], :]))
                    if dist < cfg.defect_mse_threshold:
                        drop_center_indices.add(i)
                        break

            logger.debug("Dropping %d centers near defects", len(drop_center_indices))

            new_items = torch.empty((0, dims), device=device)
            for i in range(bg_centers.shape[0]):
                if i not in drop_center_indices:
                    new_items = torch.cat((new_items, bg_centers[[i], :]))
            logger.debug("Importing %d new memory items", new_items.shape[0])
            keys = torch.cat((keys, new_items), dim=0)

            if keys.shape[0] > cfg.max_memory_size:
                keys = keys[cfg.prune_count:]
            return keys, defect_memory

        # No defect memory: simply append new centers
        keys = torch.cat((keys, bg_centers), dim=0)
        if keys.shape[0] > cfg.max_memory_size:
            keys = keys[cfg.prune_count:]
        return keys, defect_memory
