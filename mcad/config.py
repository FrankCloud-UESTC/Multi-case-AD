"""Structured configuration with dataclasses and YAML support."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    in_channels: int = 3
    feature_dim: int = 512
    key_dim: int = 512
    memory_size: int = 20


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    dataset_type: str = "ECPT"
    dataset_path: str = "./dataset/"
    class_name: str = ""
    resize_height: int = 128
    resize_width: int = 128


@dataclass
class OptimConfig:
    """Optimizer and scheduler configuration."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    scheduler: str = "cosine"  # cosine | step | none
    t_max: int = 30  # for cosine scheduler
    grad_clip_norm: float | None = None


@dataclass
class MemoryConfig:
    """Memory module configuration."""

    n_clusters: int = 10
    triplet_margin_normal: float = 0.8
    triplet_margin_defect: float = 1.0
    defect_mse_threshold: float = 0.01
    max_memory_size: int = 150
    prune_count: int = 50


@dataclass
class TrainConfig:
    """Full training configuration."""

    # Core
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Training loop
    epochs: int = 30
    batch_size: int = 1
    loss_compact: float = 0.5
    loss_separate: float = 0.5

    # Infrastructure
    device: str = "auto"  # auto | cpu | cuda:N
    amp: bool = False
    seed: int = 2023
    num_workers: int = 0

    # Output
    exp_dir: str = "exp"
    log_interval: int = 10  # log every N batches

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load config from a YAML file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        model = ModelConfig(**raw.pop("model", {}))
        data = DataConfig(**raw.pop("data", {}))
        optim = OptimConfig(**raw.pop("optim", {}))
        memory = MemoryConfig(**raw.pop("memory", {}))

        return cls(model=model, data=data, optim=optim, memory=memory, **raw)

    def to_yaml(self, path: str | Path) -> None:
        """Save config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    model_dir: str = "./model.pth"
    m_items_dir: str = "./keys.pt"
    dataset_path: str = "./"
    class_name: str = "rail_80"
    output_dir: str = "./result/"

    resize_height: int = 256
    resize_width: int = 256
    in_channels: int = 3

    feature_dim: int = 512
    key_dim: int = 512
    memory_size: int = 20

    batch_size: int = 1
    num_workers: int = 0

    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        """Load config from a YAML file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    def to_yaml(self, path: str | Path) -> None:
        """Save config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
