"""Training entry point.

Usage:
    # With default config:
    python train.py

    # With YAML config:
    python train.py --config configs/default.yaml

    # With CLI overrides:
    python train.py --epochs 50 --batch_size 4 --device cuda:0
"""

from __future__ import annotations

import argparse
import logging

from mcad.config import TrainConfig
from mcad.trainer import Trainer
from mcad.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCAD Training")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    if args.config:
        config = TrainConfig.from_yaml(args.config)
    else:
        config = TrainConfig()

    # CLI overrides
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.optim.learning_rate = args.lr
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.dataset_path is not None:
        config.data.dataset_path = args.dataset_path
    if args.amp is not None:
        config.amp = args.amp

    # Setup logging
    setup_logging(level=logging.INFO)
    logger.info("Config: %s", config)

    # Train
    trainer = Trainer(config)
    if args.resume:
        trainer.resume(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
