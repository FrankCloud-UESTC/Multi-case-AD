"""Evaluation pipeline with multi-backend inference support."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mcad.config import EvalConfig
from mcad.conv_ae import ConvAE
from mcad.data import AnomalyDataset, build_dataloader
from mcad.utils import resolve_device, get_amp_device_type

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator with multi-backend inference support.

    Supports four inference backends:
    - "pytorch": Standard PyTorch eager mode
    - "pytorch+amp": PyTorch with AMP (FP16 on GPU)
    - "onnx": ONNX Runtime GPU/CPU
    - "compile": torch.compile (Inductor backend)
    - "quantize": Dynamic INT8 quantization (CPU only)

    Args:
        config: Evaluation configuration.
    """

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.amp_device_type = get_amp_device_type(self.device)

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

        # Load model based on backend
        self._init_model()
        logger.info(
            "Evaluator ready: backend=%s, device=%s, %d samples",
            config.backend, self.device, len(dataset),
        )

    def _init_model(self) -> None:
        """Initialize model and inference backend."""
        backend = self.config.backend

        if backend == "onnx":
            self._init_onnx_backend()
        elif backend == "compile":
            self._init_compile_backend()
        elif backend == "quantize":
            self._init_quantize_backend()
        else:
            # pytorch (default) - also handles pytorch+amp
            self._init_pytorch_backend()

    def _init_pytorch_backend(self) -> None:
        """Standard PyTorch eager mode."""
        self.model = self._load_convae()
        self.m_items = self._load_keys()
        self.model.eval()
        self._infer_fn = self._infer_pytorch

    def _init_onnx_backend(self) -> None:
        """ONNX Runtime backend."""
        from mcad.export_and_inference import ONNXRuntimeInferencer, export_onnx

        # Auto-export if ONNX file doesn't exist
        onnx_path = Path(self.config.model_dir).parent / "mcad_inference.onnx"
        if not onnx_path.exists():
            logger.info("ONNX model not found, exporting...")
            onnx_path = export_onnx(
                self.config.model_dir,
                self.config.m_items_dir,
                output_dir=str(Path(self.config.model_dir).parent),
                input_height=self.config.resize_height,
                input_width=self.config.resize_width,
                device=self.config.device,
            )

        use_gpu = self.device.type == "cuda"
        self.ort_inferencer = ONNXRuntimeInferencer(onnx_path, use_gpu=use_gpu)
        self._infer_fn = self._infer_onnx

    def _init_compile_backend(self) -> None:
        """torch.compile backend with fallback for GPUs with few SMs."""
        from mcad.inference_model import build_inference_model

        self.inference_model = build_inference_model(
            self.config.model_dir, self.config.m_items_dir, self.device,
        )

        # Try Inductor first, fall back to aot_eager on old GPUs
        backends_to_try = ["inductor", "aot_eager"]
        compiled = None
        for backend in backends_to_try:
            try:
                logger.info("Compiling model with torch.compile (backend=%s)...", backend)
                compiled = torch.compile(self.inference_model, backend=backend)
                dummy = torch.randn(
                    1, 3, self.config.resize_height, self.config.resize_width, device=self.device,
                )
                with torch.no_grad():
                    compiled(dummy)
                logger.info("torch.compile warmup complete (backend=%s)", backend)
                break
            except Exception as e:
                logger.warning("torch.compile backend '%s' failed: %s", backend, e)
                compiled = None

        if compiled is not None:
            self.inference_model = compiled
        else:
            logger.warning("torch.compile unavailable on this GPU, falling back to PyTorch eager + AMP")
            self.config.backend = "pytorch"
            self.config.amp = True
            self._init_pytorch_backend()
            return

        self._infer_fn = self._infer_compile

    def _init_quantize_backend(self) -> None:
        """Dynamic INT8 quantization backend (CPU only)."""
        from mcad.export_and_inference import quantize_model_dynamic

        self.inference_model = quantize_model_dynamic(
            self.config.model_dir, self.config.m_items_dir, device=self.config.device,
        )
        self._infer_fn = self._infer_quantize

    def _load_convae(self) -> ConvAE:
        """Load ConvAE model from checkpoint."""
        checkpoint = torch.load(self.config.model_dir, map_location=self.device, weights_only=False)
        saved_config = None
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            saved_config = checkpoint["config"]

        if saved_config is not None:
            model = ConvAE(
                model_config=saved_config.model,
                memory_config=saved_config.memory,
            ).to(self.device)
        else:
            model = ConvAE().to(self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            try:
                model.load_state_dict(checkpoint)
            except RuntimeError:
                model = checkpoint.to(self.device)
        else:
            model = checkpoint.to(self.device)

        return model

    def _load_keys(self) -> list[torch.Tensor]:
        """Load memory keys."""
        m_items = torch.load(self.config.m_items_dir, map_location=self.device, weights_only=False)
        return [m_items] if m_items.dim() == 2 else m_items

    # -- Inference functions per backend --

    def _infer_pytorch(self, imgs: torch.Tensor) -> np.ndarray:
        """PyTorch eager inference with optional AMP."""
        use_amp = self.config.amp and self.device.type == "cuda"
        with torch.no_grad(), torch.amp.autocast(self.amp_device_type, enabled=use_amp):
            heatmap, _, _, _ = self.model(imgs, self.m_items, train=False, has_defect=False)
        return heatmap[0, :, :, 0].detach().cpu().numpy()

    def _infer_onnx(self, imgs: torch.Tensor) -> np.ndarray:
        """ONNX Runtime inference."""
        heatmap = self.ort_inferencer.infer(imgs.cpu().numpy())
        return heatmap[0, :, :, 0]

    def _infer_compile(self, imgs: torch.Tensor) -> np.ndarray:
        """torch.compile inference."""
        imgs = imgs.to(self.device)
        use_amp = self.config.amp and self.device.type == "cuda"
        with torch.no_grad(), torch.amp.autocast(self.amp_device_type, enabled=use_amp):
            heatmap = self.inference_model(imgs)
        return heatmap[0, :, :, 0].detach().cpu().numpy()

    def _infer_quantize(self, imgs: torch.Tensor) -> np.ndarray:
        """Quantized model inference (CPU)."""
        with torch.no_grad():
            heatmap = self.inference_model(imgs.cpu())
        return heatmap[0, :, :, 0].detach().numpy()

    def evaluate(self) -> None:
        """Run evaluation and save heatmap visualizations."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Evaluating %d samples (backend=%s)", len(self.dataloader), self.config.backend)

        for imgs, label, name in tqdm(self.dataloader, desc="Evaluating"):
            heatmap = self._infer_fn(imgs)

            # Convert image for visualization
            img_np = imgs[0].cpu().detach().numpy()
            img_np = np.moveaxis(img_np, 0, 2)  # CHW -> HWC
            img_np = (img_np + 1) * 127 if img_np.min() < 0 else img_np * 255

            # Resize heatmap to original image dimensions
            mask = cv2.resize(heatmap, (self.config.resize_width, self.config.resize_height))

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
