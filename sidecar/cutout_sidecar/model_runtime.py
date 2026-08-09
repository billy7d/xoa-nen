from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .models import list_model_manifests

try:
    import onnxruntime as ort  # type: ignore
except ImportError:  # pragma: no cover - explicit fallback in source-only development.
    ort = None


def _resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb, "RGB").resize(size, Image.Resampling.BILINEAR))


def _resize_float(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(values.astype(np.float32), "F").resize(size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def _sigmoid_if_needed(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    if float(np.min(values)) < -0.001 or float(np.max(values)) > 1.001:
        values = 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))
    return np.clip(values, 0.0, 1.0)


class LocalModelRuntime:
    """Audited ONNX-only runtime. It never imports or executes model repository code."""

    def __init__(self, model_directory: Path) -> None:
        self.model_directory = model_directory

    def _ready(self, role: str) -> dict[str, Any] | None:
        return next(
            (
                model
                for model in list_model_manifests(self.model_directory)
                if model.get("installed") and model.get("role") == role
            ),
            None,
        )

    @staticmethod
    def _providers(manifest: dict[str, Any]) -> list[str]:
        if ort is None:
            return []
        available = set(ort.get_available_providers())
        qualified = list(manifest.get("qualified_backends") or [])
        preferred = [
            provider
            for provider in ("CoreMLExecutionProvider", "CPUExecutionProvider")
            if provider in available and provider in qualified
        ]
        return preferred

    @staticmethod
    def _artifact_path(manifest: dict[str, Any]) -> Path:
        artifacts = manifest.get("artifacts") or []
        if not artifacts and manifest.get("weight_filename"):
            artifacts = [{"filename": manifest["weight_filename"]}]
        if not artifacts:
            raise ValueError("Model manifest không có ONNX artifact")
        return Path(manifest["install_path"]) / artifacts[0]["filename"]

    def semantic_proposal(self, rgb: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
        started = time.perf_counter()
        manifest = self._ready("base_alpha_proposal")
        if manifest is None:
            return None, {"status": "model_not_installed"}
        providers = self._providers(manifest)
        if ort is None or not providers:
            return None, {"status": "runtime_or_qualified_backend_unavailable"}
        try:
            session = ort.InferenceSession(str(self._artifact_path(manifest)), providers=providers)
            input_size = manifest.get("input_size", [1024, 1024])
            width, height = int(input_size[0]), int(input_size[1])
            resized = _resize_rgb(rgb, (width, height)).astype(np.float32) / 255.0
            mean = np.asarray(manifest.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
            std = np.asarray(manifest.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
            tensor = ((resized - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
            input_name = str(manifest.get("input_name") or session.get_inputs()[0].name)
            output_name = manifest.get("output_name")
            outputs = session.run([output_name] if output_name else None, {input_name: tensor})
            proposal = np.asarray(outputs[0], dtype=np.float32).squeeze()
            if proposal.ndim != 2:
                raise ValueError(f"Semantic output shape không hỗ trợ: {proposal.shape}")
            proposal = _sigmoid_if_needed(proposal)
            proposal = _resize_float(proposal, (rgb.shape[1], rgb.shape[0]))
            return proposal, {
                "status": "ok",
                "model_id": manifest.get("model_id"),
                "revision": manifest.get("revision"),
                "backend": session.get_providers()[0],
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        except Exception as error:
            return None, {
                "status": "inference_failed",
                "error": f"{type(error).__name__}: {error}",
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }

    def refine_unknown(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        source_alpha: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run ViTMatte only on a boundary ROI; clamp known regions afterwards."""
        started = time.perf_counter()
        manifest = self._ready("roi_matting")
        if manifest is None:
            return alpha, {"status": "model_not_installed"}
        providers = self._providers(manifest)
        if ort is None or not providers:
            return alpha, {"status": "runtime_or_qualified_backend_unavailable"}
        foreground = alpha >= np.minimum(source_alpha, 0.95)
        background = alpha <= 0.02
        # A compact morphological band without SciPy/OpenCV keeps the adapter
        # deterministic and makes sure the model cannot overwrite known pixels.
        fg_image = Image.fromarray(foreground.astype(np.uint8) * 255, "L")
        bg_image = Image.fromarray(background.astype(np.uint8) * 255, "L")
        eroded_fg = np.asarray(fg_image.filter(ImageFilter.MinFilter(9)), dtype=np.uint8) > 0
        eroded_bg = np.asarray(bg_image.filter(ImageFilter.MinFilter(9)), dtype=np.uint8) > 0
        unknown = ~(eroded_fg | eroded_bg)
        ys, xs = np.nonzero(unknown)
        if not xs.size:
            return alpha, {"status": "no_unknown_roi"}
        margin = 32
        x0, x1 = max(0, int(xs.min()) - margin), min(rgb.shape[1], int(xs.max()) + margin + 1)
        y0, y1 = max(0, int(ys.min()) - margin), min(rgb.shape[0], int(ys.max()) + margin + 1)
        try:
            session = ort.InferenceSession(str(self._artifact_path(manifest)), providers=providers)
            input_size = manifest.get("input_size", [512, 512])
            width, height = int(input_size[0]), int(input_size[1])
            crop_rgb = _resize_rgb(rgb[y0:y1, x0:x1], (width, height)).astype(np.float32) / 255.0
            trimap = np.full(alpha[y0:y1, x0:x1].shape, 0.5, dtype=np.float32)
            trimap[eroded_bg[y0:y1, x0:x1]] = 0.0
            trimap[eroded_fg[y0:y1, x0:x1]] = 1.0
            trimap = _resize_float(trimap, (width, height))
            mean = np.asarray(manifest.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
            std = np.asarray(manifest.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
            image_tensor = ((crop_rgb - mean) / std).transpose(2, 0, 1)
            tensor = np.concatenate((image_tensor, trimap[None]), axis=0)[None].astype(np.float32)
            input_name = str(manifest.get("input_name") or session.get_inputs()[0].name)
            output_name = manifest.get("output_name")
            outputs = session.run([output_name] if output_name else None, {input_name: tensor})
            matte = _sigmoid_if_needed(np.asarray(outputs[0], dtype=np.float32).squeeze())
            matte = _resize_float(matte, (x1 - x0, y1 - y0))
            result = alpha.copy()
            local_unknown = unknown[y0:y1, x0:x1]
            view = result[y0:y1, x0:x1]
            view[local_unknown] = matte[local_unknown] * source_alpha[y0:y1, x0:x1][local_unknown]
            result[background] = 0.0
            result[foreground] = source_alpha[foreground]
            return np.minimum(np.clip(result, 0.0, 1.0), source_alpha), {
                "status": "ok",
                "model_id": manifest.get("model_id"),
                "revision": manifest.get("revision"),
                "backend": session.get_providers()[0],
                "roi": [x0, y0, x1, y1],
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        except Exception as error:
            return alpha, {
                "status": "inference_failed",
                "error": f"{type(error).__name__}: {error}",
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
