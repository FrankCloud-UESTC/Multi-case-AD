"""Evaluation entry point with multi-backend inference support.

Usage:
    # Standard PyTorch inference
    python Evaluate.py --model_dir model.pth --m_items_dir keys.pt

    # PyTorch + AMP (FP16 on GPU)
    python Evaluate.py --model_dir model.pth --m_items_dir keys.pt --amp

    # ONNX Runtime inference
    python Evaluate.py --model_dir model.pth --m_items_dir keys.pt --backend onnx

    # torch.compile inference
    python Evaluate.py --model_dir model.pth --m_items_dir keys.pt --backend compile

    # Dynamic INT8 quantization (CPU)
    python Evaluate.py --model_dir model.pth --m_items_dir keys.pt --backend quantize
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
    parser.add_argument("--amp", action="store_true", default=None, help="Enable AMP (FP16) inference")
    parser.add_argument(
        "--backend", type=str, default=None,
        choices=["pytorch", "onnx", "compile", "quantize"],
        help="Inference backend",
    )
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
    if args.amp is not None:
        config.amp = args.amp
    if args.backend is not None:
        config.backend = args.backend

    # Setup logging
    setup_logging(level=logging.INFO)

    # Evaluate
    evaluator = Evaluator(config)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
