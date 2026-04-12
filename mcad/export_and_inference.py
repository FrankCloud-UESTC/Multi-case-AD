"""Export and inference pipelines: ONNX export, ONNX Runtime, torch.compile, quantization."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from mcad.config import EvalConfig
from mcad.inference_model import InferenceModel, build_inference_model
from mcad.utils import resolve_device, get_amp_device_type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_onnx(
    checkpoint_path: str,
    keys_path: str,
    output_dir: str = "./export",
    input_height: int = 128,
    input_width: int = 128,
    batch_size: int = 1,
    opset_version: int = 17,
    device: str = "auto",
) -> str:
    """Export the inference model to ONNX format.

    Args:
        checkpoint_path: Path to training checkpoint.
        keys_path: Path to saved memory keys.
        output_dir: Directory to save the ONNX model.
        input_height: Input image height.
        input_width: Input image width.
        batch_size: Batch size for the exported model.
        opset_version: ONNX opset version.
        device: Device specification.

    Returns:
        Path to the saved ONNX model.
    """
    dev = resolve_device(device)
    model = build_inference_model(checkpoint_path, keys_path, dev)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "mcad_inference.onnx"

    dummy_input = torch.randn(batch_size, 3, input_height, input_width, device=dev)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=opset_version,
        input_names=["images"],
        output_names=["heatmap"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "heatmap": {0: "batch_size"},
        },
    )

    # Validate the exported model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    logger.info("ONNX model exported to %s (opset=%d)", onnx_path, opset_version)
    return str(onnx_path)


# ---------------------------------------------------------------------------
# ONNX Runtime Inference
# ---------------------------------------------------------------------------

class ONNXRuntimeInferencer:
    """Inference using ONNX Runtime with GPU/CPU execution providers.

    Provides significant speedup over PyTorch eager mode through:
    - Operator fusion
    - Optimized kernel selection
    - Reduced memory overhead

    Args:
        onnx_path: Path to the ONNX model file.
        use_gpu: Whether to use CUDA execution provider.
        device_id: GPU device ID.
    """

    def __init__(self, onnx_path: str, use_gpu: bool = True, device_id: int = 0) -> None:
        self.onnx_path = onnx_path

        providers: list[str | tuple] = []
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        active_provider = self.session.get_providers()[0]
        logger.info("ONNX Runtime session created (provider=%s)", active_provider)

    def infer(self, images: np.ndarray) -> np.ndarray:
        """Run inference on a batch of images.

        Args:
            images: Input images as numpy array (B, C, H, W), float32.

        Returns:
            Anomaly heatmaps as numpy array (B, H', W', 1).
        """
        return self.session.run(None, {self.input_name: images})[0]


# ---------------------------------------------------------------------------
# torch.compile Inference
# ---------------------------------------------------------------------------

class TorchCompileInferencer:
    """Inference using torch.compile for kernel fusion and optimization.

    torch.compile (PyTorch 2.0+) compiles the model into optimized kernels
    using TorchDynamo and Inductor backend. Zero code changes needed.

    Args:
        checkpoint_path: Path to training checkpoint.
        keys_path: Path to saved memory keys.
        device: Device specification.
        backend: torch.compile backend (default: "inductor").
        mode: Compilation mode - "default", "reduce-overhead", or "max-autotune".
    """

    def __init__(
        self,
        checkpoint_path: str,
        keys_path: str,
        device: str = "auto",
        backend: str = "inductor",
        mode: str = "default",
    ) -> None:
        self.device = resolve_device(device)
        self.model = build_inference_model(checkpoint_path, keys_path, self.device)

        # Try compile with fallback for GPUs with few SMs
        compile_kwargs: dict = {}
        if mode != "default":
            compile_kwargs["mode"] = mode

        backends_to_try = [backend, "aot_eager"] if backend != "aot_eager" else ["aot_eager"]
        compiled = None
        for b in backends_to_try:
            try:
                logger.info("Compiling model with torch.compile (backend=%s, mode=%s)...", b, mode)
                compiled = torch.compile(self.model, backend=b, **compile_kwargs)

                # Warmup
                dummy = torch.randn(1, 3, 128, 128, device=self.device)
                with torch.no_grad():
                    with torch.amp.autocast(get_amp_device_type(self.device), enabled=self.device.type == "cuda"):
                        compiled(dummy)
                logger.info("torch.compile warmup complete (backend=%s)", b)
                break
            except Exception as e:
                logger.warning("torch.compile backend '%s' failed: %s", b, e)
                compiled = None

        if compiled is not None:
            self.model = compiled
        else:
            logger.warning("torch.compile unavailable, using eager mode")


    def infer(self, images: torch.Tensor) -> torch.Tensor:
        """Run inference on a batch of images.

        Args:
            images: Input images tensor (B, C, H, W).

        Returns:
            Anomaly heatmaps tensor (B, H', W', 1).
        """
        images = images.to(self.device)
        with torch.no_grad():
            with torch.amp.autocast(get_amp_device_type(self.device), enabled=self.device.type == "cuda"):
                return self.model(images)


# ---------------------------------------------------------------------------
# Post-Training Quantization (PTQ) - Dynamic
# ---------------------------------------------------------------------------

def quantize_model_dynamic(
    checkpoint_path: str,
    keys_path: str,
    device: str = "auto",
) -> InferenceModel:
    """Apply dynamic post-training quantization (INT8) to the encoder.

    Dynamic quantization converts weight matrices to INT8 and quantizes
    activations dynamically during inference. Works on CPU only.

    Args:
        checkpoint_path: Path to training checkpoint.
        keys_path: Path to saved memory keys.
        device: Device specification (must be cpu for quantization).

    Returns:
        Quantized InferenceModel.
    """
    dev = resolve_device(device)
    if dev.type == "cuda":
        logger.warning("Dynamic quantization requires CPU, switching to CPU")
        dev = torch.device("cpu")

    model = build_inference_model(checkpoint_path, keys_path, dev)

    # Quantize the encoder (the heavy part)
    model.encoder = torch.ao.quantization.quantize_dynamic(
        model.encoder,
        {torch.nn.Conv2d},
        dtype=torch.qint8,
    )

    logger.info("Applied dynamic INT8 quantization to encoder")
    return model


# ---------------------------------------------------------------------------
# Benchmarking Utility
# ---------------------------------------------------------------------------

def benchmark(
    checkpoint_path: str,
    keys_path: str,
    onnx_path: str | None = None,
    input_height: int = 128,
    input_width: int = 128,
    batch_size: int = 1,
    num_warmup: int = 5,
    num_iterations: int = 50,
    device: str = "auto",
) -> dict[str, float]:
    """Benchmark different inference backends.

    Args:
        checkpoint_path: Path to training checkpoint.
        keys_path: Path to saved memory keys.
        onnx_path: Path to ONNX model (if already exported).
        input_height: Input image height.
        input_width: Input image width.
        batch_size: Batch size for benchmarking.
        num_warmup: Number of warmup iterations.
        num_iterations: Number of timed iterations.
        device: Device specification.

    Returns:
        Dictionary mapping backend name to average latency (ms).
    """
    dev = resolve_device(device)
    results: dict[str, float] = {}

    # --- PyTorch eager ---
    logger.info("Benchmarking PyTorch eager...")
    model_eager = build_inference_model(checkpoint_path, keys_path, dev)
    dummy = torch.randn(batch_size, 3, input_height, input_width, device=dev)

    for _ in range(num_warmup):
        with torch.no_grad():
            model_eager(dummy)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            model_eager(dummy)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iterations * 1000
    results["pytorch_eager"] = elapsed
    logger.info("  PyTorch eager: %.2f ms", elapsed)

    # --- PyTorch eager + AMP ---
    if dev.type == "cuda":
        logger.info("Benchmarking PyTorch eager + AMP...")
        for _ in range(num_warmup):
            with torch.no_grad(), torch.amp.autocast("cuda"):
                model_eager(dummy)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iterations):
            with torch.no_grad(), torch.amp.autocast("cuda"):
                model_eager(dummy)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iterations * 1000
        results["pytorch_amp"] = elapsed
        logger.info("  PyTorch AMP: %.2f ms", elapsed)

    # --- torch.compile ---
    try:
        logger.info("Benchmarking torch.compile...")
        for compile_backend in ["inductor", "aot_eager"]:
            try:
                model_compiled = torch.compile(model_eager, backend=compile_backend)
                extra_warmup = 3 if compile_backend == "inductor" else 1
                for _ in range(num_warmup + extra_warmup):
                    with torch.no_grad():
                        model_compiled(dummy)
                if dev.type == "cuda":
                    torch.cuda.synchronize()

                start = time.perf_counter()
                for _ in range(num_iterations):
                    with torch.no_grad():
                        model_compiled(dummy)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) / num_iterations * 1000
                results[f"torch_compile_{compile_backend}"] = elapsed
                logger.info("  torch.compile (%s): %.2f ms", compile_backend, elapsed)
                break  # Use the first backend that works
            except Exception as e:
                logger.warning("  torch.compile backend '%s' failed: %s", compile_backend, e)
    except Exception as e:
        logger.warning("  torch.compile benchmark failed: %s", e)

    # --- ONNX Runtime ---
    if onnx_path is not None and Path(onnx_path).exists():
        logger.info("Benchmarking ONNX Runtime...")
        use_gpu = dev.type == "cuda"
        ort_inferencer = ONNXRuntimeInferencer(onnx_path, use_gpu=use_gpu)
        dummy_np = dummy.cpu().numpy()

        for _ in range(num_warmup):
            ort_inferencer.infer(dummy_np)

        start = time.perf_counter()
        for _ in range(num_iterations):
            ort_inferencer.infer(dummy_np)
        elapsed = (time.perf_counter() - start) / num_iterations * 1000
        results["onnx_runtime"] = elapsed
        logger.info("  ONNX Runtime: %.2f ms", elapsed)

    # --- Summary ---
    logger.info("=" * 50)
    logger.info("Benchmark Results (batch_size=%d, %dx%d):", batch_size, input_height, input_width)
    baseline = results.get("pytorch_eager", 1.0)
    for name, latency in sorted(results.items(), key=lambda x: x[1]):
        speedup = baseline / latency
        logger.info("  %-20s %8.2f ms  (%.2fx)", name, latency, speedup)

    return results
