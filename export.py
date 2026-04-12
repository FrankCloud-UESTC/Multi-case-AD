"""Export and benchmark entry point.

Usage:
    # Export to ONNX
    python export.py --checkpoint exp/ECPT/log/checkpoint_best.pth --keys exp/ECPT/log/keys.pt

    # Benchmark all backends
    python export.py --benchmark --checkpoint exp/ECPT/log/checkpoint_best.pth --keys exp/ECPT/log/keys.pt

    # Export with custom input size
    python export.py --checkpoint model.pth --keys keys.pt --height 256 --width 256
"""

from __future__ import annotations

import argparse
import logging

from mcad.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCAD Export & Benchmark")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to training checkpoint")
    parser.add_argument("--keys", type=str, required=True, help="Path to memory keys (.pt)")
    parser.add_argument("--output_dir", type=str, default="./export", help="Output directory for ONNX")
    parser.add_argument("--height", type=int, default=128, help="Input height")
    parser.add_argument("--width", type=int, default=128, help="Input width")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--num_iterations", type=int, default=50, help="Benchmark iterations")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    # Export ONNX
    from mcad.export_and_inference import export_onnx

    onnx_path = export_onnx(
        checkpoint_path=args.checkpoint,
        keys_path=args.keys,
        output_dir=args.output_dir,
        input_height=args.height,
        input_width=args.width,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Benchmark
    if args.benchmark:
        from mcad.export_and_inference import benchmark

        benchmark(
            checkpoint_path=args.checkpoint,
            keys_path=args.keys,
            onnx_path=onnx_path,
            input_height=args.height,
            input_width=args.width,
            batch_size=args.batch_size,
            num_iterations=args.num_iterations,
            device=args.device,
        )


if __name__ == "__main__":
    main()
