"""Evaluation entry point.

Usage:
    # With YAML config:
    python evaluate.py --config configs/eval.yaml

    # With CLI overrides:
    python evaluate.py --model_dir exp/ECPT/log/checkpoint_best.pth --m_items_dir exp/ECPT/log/keys.pt
"""

from __future__ import annotations

import argparse
import logging

from mcad.config import EvalConfig
from mcad.evaluator import Evaluator
from mcad.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCAD Evaluation")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--m_items_dir", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    if args.config:
        config = EvalConfig.from_yaml(args.config)
    else:
        config = EvalConfig()

    # CLI overrides
    if args.model_dir is not None:
        config.model_dir = args.model_dir
    if args.m_items_dir is not None:
        config.m_items_dir = args.m_items_dir
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.device is not None:
        config.device = args.device

    # Setup logging
    setup_logging(level=logging.INFO)

    # Evaluate
    evaluator = Evaluator(config)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
