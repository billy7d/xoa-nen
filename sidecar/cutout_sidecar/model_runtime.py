from __future__ import annotations

import threading
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


_SESSION_CACHE: dict[tuple[str, tuple[str, ...], str, str], Any] = {}
_SESSION_CACHE_LOCK = threading.RLock()


def _resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb, "RGB").resize(size, Image.Resampling.BILINEAR))


def _resize_float(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(values.astype(np.float32), "F").resize(size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def _letterbox_rgb(rgb: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Giữ tỷ lệ ảnh semantic; trả cả vùng ảnh thật để inverse-map chính xác."""
    target_width, target_height = size
    scale = min(target_width / rgb.shape[1], target_height / rgb.shape[0])
    width = max(1, min(target_width, round(rgb.shape[1] * scale)))
    height = max(1, min(target_height, round(rgb.shape[0] * scale)))
    x0 = (target_width - width) // 2
    y0 = (target_height - height) // 2
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    canvas[y0 : y0 + height, x0 : x0 + width] = _resize_rgb(rgb, (width, height))
    return canvas, (x0, y0, width, height)


def _hann_weight(height: int, width: int) -> np.ndarray:
    """Tạo trọng số ghép tile, tránh seam tại vùng chồng lấn ViTMatte."""
    if height <= 1 or width <= 1:
        return np.ones((height, width), dtype=np.float32)
    return np.outer(np.hanning(height), np.hanning(width)).astype(np.float32) + 1e-3


def _sigmoid_if_needed(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    if float(np.min(values)) < -0.001 or float(np.max(values)) > 1.001:
        values = 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))
    return np.clip(values, 0.0, 1.0)


def _input_size(manifest: dict[str, Any], default: tuple[int, int]) -> tuple[int, int]:
    configured = manifest.get("input_size", list(default))
    if not isinstance(configured, (list, tuple)) or len(configured) != 2:
        raise ValueError("Model manifest input_size phải có dạng [width, height]")
    width, height = (int(configured[0]), int(configured[1]))
    if width <= 0 or height <= 0 or width > 8192 or height > 8192:
        raise ValueError("Model manifest input_size nằm ngoài giới hạn an toàn")
    return width, height


def _normalise_image(rgb: np.ndarray, size: tuple[int, int], manifest: dict[str, Any]) -> np.ndarray:
    resized = _resize_rgb(rgb, size).astype(np.float32) / 255.0
    if str(manifest.get("color_order", "RGB")).upper() == "BGR":
        resized = resized[..., ::-1]
    normalization = str(manifest.get("normalization", "imagenet")).lower()
    if normalization in {"none", "zero_one"}:
        return resized
    if normalization in {"minus_one_one", "-1_1"}:
        return resized * 2.0 - 1.0
    mean = np.asarray(manifest.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.asarray(manifest.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0):
        raise ValueError("Model manifest mean/std không hợp lệ")
    return (resized - mean) / std


def _tensor(values: np.ndarray, layout: str) -> np.ndarray:
    layout = layout.upper()
    if values.ndim == 2:
        values = values[..., None]
    if values.ndim != 3:
        raise ValueError(f"Tensor đầu vào phải là HWC, nhận {values.shape}")
    if layout == "NHWC":
        return values[None].astype(np.float32)
    if layout == "NCHW":
        return values.transpose(2, 0, 1)[None].astype(np.float32)
    raise ValueError(f"Layout đầu vào không được hỗ trợ: {layout}")


def _layout_from_input(manifest: dict[str, Any], model_input: Any, channels: int = 3) -> str:
    configured = str(manifest.get("input_layout", "AUTO")).upper()
    if configured in {"NCHW", "NHWC"}:
        return configured
    shape = list(getattr(model_input, "shape", []) or [])
    if len(shape) == 4:
        if shape[1] == channels or shape[1] is None:
            return "NCHW"
        if shape[-1] == channels:
            return "NHWC"
    return "NCHW"


def _mask_output(values: Any, manifest: dict[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    layout = str(manifest.get("output_layout", "AUTO")).upper()
    channel = int(manifest.get("output_channel", 0))
    if channel < 0:
        raise ValueError("Model manifest output_channel không hợp lệ")
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"Output mask chỉ hỗ trợ batch=1, nhận {array.shape}")
        auto_nhwc = layout == "AUTO" and array.shape[-1] <= 16 and array.shape[1] > 16
        if layout == "NHWC" or auto_nhwc:
            if channel >= array.shape[-1]:
                raise ValueError(f"Output channel vượt kích thước {array.shape}")
            array = array[0, :, :, channel]
        else:
            if channel >= array.shape[1]:
                raise ValueError(f"Output channel vượt kích thước {array.shape}")
            array = array[0, channel, :, :]
    elif array.ndim == 3:
        auto_nhwc = layout == "AUTO" and array.shape[-1] <= 16 and array.shape[0] > 16
        if array.shape[0] == 1:
            array = array[0]
        elif layout == "NHWC" or auto_nhwc:
            if channel >= array.shape[-1]:
                raise ValueError(f"Output channel vượt kích thước {array.shape}")
            array = array[:, :, channel]
        else:
            if channel >= array.shape[0]:
                raise ValueError(f"Output channel vượt kích thước {array.shape}")
            array = array[channel, :, :]
    if array.ndim != 2:
        raise ValueError(f"Output mask phải là ma trận 2D, nhận {array.shape}")
    activation = str(manifest.get("output_activation", "auto")).lower()
    if activation == "sigmoid":
        array = 1.0 / (1.0 + np.exp(-np.clip(array, -30.0, 30.0)))
    elif activation not in {"auto", "identity", "linear"}:
        raise ValueError(f"Output activation không được hỗ trợ: {activation}")
    elif activation == "auto":
        array = _sigmoid_if_needed(array)
    array = np.clip(array, 0.0, 1.0)
    if str(manifest.get("output_semantics", "foreground")).lower() == "background":
        array = 1.0 - array
    return array.astype(np.float32)


def _prompt_points(
    points: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    """Chuẩn hoá point prompt, không để input giao diện gây lỗi inference."""
    result: list[tuple[float, float]] = []
    for item in points or []:
        if not isinstance(item, dict):
            continue
        try:
            x = float(item.get("x"))
            y = float(item.get("y"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            result.append((min(width - 1.0, max(0.0, x)), min(height - 1.0, max(0.0, y))))
    return result


def _focus_tile_bounds(
    width: int,
    height: int,
    point: tuple[float, float],
    input_size: tuple[int, int],
    vertical_bias: float,
) -> tuple[int, int, int, int]:
    """Tạo tile chồng lấn quanh click để Lite 512 nhìn rõ chi tiết mảnh."""
    tile_width = min(width, max(input_size[0], round(width * 0.80)))
    tile_height = min(height, max(input_size[1], round(height * 0.80)))
    center_x = point[0]
    center_y = point[1] + vertical_bias * tile_height
    x0 = min(max(0, int(round(center_x - tile_width / 2))), max(0, width - tile_width))
    y0 = min(max(0, int(round(center_y - tile_height / 2))), max(0, height - tile_height))
    return x0, y0, x0 + tile_width, y0 + tile_height


class LocalModelRuntime:
    """Audited ONNX-only runtime. It never imports or executes model repository code."""

    def __init__(self, model_directory: Path) -> None:
        self.model_directory = model_directory

    def _ready(self, role: str) -> dict[str, Any] | None:
        candidates = [
            model
            for model in list_model_manifests(self.model_directory)
            if model.get("installed") and model.get("role") == role
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda model: (int(model.get("priority", 0)), str(model.get("model_id", ""))),
        )

    @staticmethod
    def _providers(manifest: dict[str, Any]) -> list[str]:
        if ort is None:
            return []
        available = set(ort.get_available_providers())
        qualified = list(manifest.get("qualified_backends") or [])
        preferred = [
            provider
            for provider in (
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            )
            if provider in available and provider in qualified
        ]
        return preferred

    @staticmethod
    def _artifact_path(manifest: dict[str, Any], artifact_role: str | None = None) -> Path:
        artifacts = manifest.get("artifacts") or []
        if not artifacts and manifest.get("weight_filename"):
            artifacts = [{"filename": manifest["weight_filename"]}]
        if not artifacts:
            raise ValueError("Model manifest không có ONNX artifact")
        selected = artifacts[0]
        if artifact_role:
            selected = next(
                (
                    artifact
                    for artifact in artifacts
                    if artifact.get("role") == artifact_role
                    or artifact.get("artifact_role") == artifact_role
                ),
                selected,
            )
        return (Path(manifest["install_path"]) / str(selected["filename"])).resolve()

    @classmethod
    def _session(cls, manifest: dict[str, Any], artifact_role: str | None = None) -> Any:
        if ort is None:
            raise RuntimeError("onnxruntime chưa được cài")
        path = cls._artifact_path(manifest, artifact_role)
        providers = tuple(cls._providers(manifest))
        if not providers:
            raise RuntimeError("Không có backend ONNX Runtime đã qualification")
        key = (
            str(path),
            providers,
            str(manifest.get("revision", "")),
            str(manifest.get("adapter", "")),
        )
        with _SESSION_CACHE_LOCK:
            session = _SESSION_CACHE.get(key)
            if session is None:
                # DirectML yêu cầu chạy tuần tự và tắt memory-pattern để tránh lỗi/VRAM dư.
                options = ort.SessionOptions() if hasattr(ort, "SessionOptions") else None
                if options is not None and "DmlExecutionProvider" in providers:
                    options.enable_mem_pattern = False
                    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                if options is None:
                    session = ort.InferenceSession(str(path), providers=list(providers))
                else:
                    session = ort.InferenceSession(
                        str(path), sess_options=options, providers=list(providers)
                    )
                _SESSION_CACHE[key] = session
            return session

    @staticmethod
    def _run_mask(session: Any, inputs: dict[str, np.ndarray], manifest: dict[str, Any]) -> np.ndarray:
        output_name = manifest.get("output_name")
        outputs = session.run([str(output_name)] if output_name else None, inputs)
        output_index = int(manifest.get("output_index", 0))
        if output_index < 0 or output_index >= len(outputs):
            raise ValueError(f"Output index không hợp lệ: {output_index}")
        return _mask_output(outputs[output_index], manifest)

    @staticmethod
    def _image_input(
        rgb: np.ndarray, manifest: dict[str, Any], model_input: Any, default_size: tuple[int, int]
    ) -> tuple[np.ndarray, str]:
        size = _input_size(manifest, default_size)
        layout = _layout_from_input(manifest, model_input, channels=3)
        values = _normalise_image(rgb, size, manifest)
        return _tensor(values, layout), layout

    @staticmethod
    def _letterboxed_image_input(
        rgb: np.ndarray, manifest: dict[str, Any], model_input: Any, default_size: tuple[int, int]
    ) -> tuple[np.ndarray, str, tuple[int, int, int, int]]:
        size = _input_size(manifest, default_size)
        layout = _layout_from_input(manifest, model_input, channels=3)
        letterboxed, mapping = _letterbox_rgb(rgb, size)
        values = _normalise_image(letterboxed, size, manifest)
        return _tensor(values, layout), layout, mapping

    @staticmethod
    def _input_name(manifest: dict[str, Any], model_inputs: list[Any], key: str, index: int = 0) -> str:
        configured = manifest.get(key)
        names = [str(item.name) for item in model_inputs]
        if configured:
            configured = str(configured)
            if configured not in names:
                raise ValueError(f"Input {configured} không có trong model: {names}")
            return configured
        if index >= len(names):
            raise ValueError(f"Model thiếu input thứ {index}: {names}")
        return names[index]

    @staticmethod
    def _image_output(values: Any, manifest: dict[str, Any]) -> np.ndarray:
        """Chuẩn hoá tensor SR về RGB thẳng 0..255, không áp sigmoid lên ảnh màu."""
        array = np.asarray(values, dtype=np.float32)
        layout = str(manifest.get("output_layout", "AUTO")).upper()
        if array.ndim == 4:
            if array.shape[0] != 1:
                raise ValueError(f"Output SR chỉ hỗ trợ batch=1, nhận {array.shape}")
            auto_nhwc = layout == "AUTO" and array.shape[-1] in {1, 3, 4}
            array = array[0] if (layout == "NHWC" or auto_nhwc) else array[0].transpose(1, 2, 0)
        elif array.ndim == 3 and layout == "NCHW" and array.shape[0] in {1, 3, 4}:
            array = array.transpose(1, 2, 0)
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise ValueError(f"Output SR phải là ảnh RGB/RGBA, nhận {array.shape}")
        array = array[:, :, :3]
        output_range = str(manifest.get("output_range", "zero_one")).lower()
        if output_range in {"minus_one_one", "-1_1"}:
            array = (array + 1.0) * 0.5
        elif output_range not in {"zero_one", "0_1"}:
            raise ValueError(f"output_range SR không hỗ trợ: {output_range}")
        if str(manifest.get("color_order", "RGB")).upper() == "BGR":
            array = array[:, :, ::-1]
        return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)

    def inpaint_rgb(
        self, rgb: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Run a signed LaMa-compatible image-inpainting pack on a local ROI.

        Different ONNX exports expose either one 4-channel RGB+mask tensor or
        separate image/mask inputs.  The manifest makes that distinction
        explicit while this adapter never loads model repository code.
        """
        started = time.perf_counter()
        manifest = self._ready("image_inpainting")
        if manifest is None:
            return None, {"status": "model_not_installed", "role": "image_inpainting"}
        providers = self._providers(manifest)
        if ort is None or not providers:
            return None, {"status": "runtime_or_qualified_backend_unavailable", "role": "image_inpainting"}
        try:
            adapter = str(manifest.get("adapter", "")).lower()
            if adapter not in {"lama-inpaint-v1", "lama", "generic-inpaint", "image-inpainting-v1"}:
                return None, {"status": "unsupported_inpaint_adapter", "adapter": adapter}
            if rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2]:
                return None, {"status": "invalid_inpaint_input"}
            ys, xs = np.nonzero(mask > 0)
            if not xs.size:
                return rgb.copy(), {"status": "empty_mask"}
            padding = max(32, min(256, max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)) * 2))
            x0, y0 = max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding)
            x1, y1 = min(rgb.shape[1], int(xs.max()) + 1 + padding), min(rgb.shape[0], int(ys.max()) + 1 + padding)
            roi, roi_mask = rgb[y0:y1, x0:x1], (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
            session = self._session(manifest)
            model_inputs = list(session.get_inputs())
            if not model_inputs:
                raise ValueError("Model inpainting không có input")
            size = _input_size(manifest, (512, 512))
            image_name = self._input_name(manifest, model_inputs, "input_name")
            image_layout = _layout_from_input(manifest, model_inputs[0], channels=3)
            image = _normalise_image(roi, size, manifest)
            mask_image = np.asarray(
                Image.fromarray(roi_mask * 255, "L").resize(size, Image.Resampling.NEAREST), dtype=np.float32
            ) / 255.0
            if str(manifest.get("mask_semantics", "masked_one")).lower() == "valid_one":
                mask_image = 1.0 - mask_image
            if len(model_inputs) >= 2 or manifest.get("mask_input_name"):
                mask_name = self._input_name(manifest, model_inputs, "mask_input_name", index=1)
                mask_layout = _layout_from_input(manifest, model_inputs[1], channels=1)
                inputs = {image_name: _tensor(image, image_layout), mask_name: _tensor(mask_image, mask_layout)}
            else:
                # One-input LaMa exports conventionally concatenate RGB+mask.
                layout = _layout_from_input(manifest, model_inputs[0], channels=4)
                inputs = {image_name: _tensor(np.dstack((image, mask_image)), layout)}
            output_name = manifest.get("output_name")
            outputs = session.run([str(output_name)] if output_name else None, inputs)
            output_index = int(manifest.get("output_index", 0))
            if output_index < 0 or output_index >= len(outputs):
                raise ValueError("Output index inpainting không hợp lệ")
            generated = self._image_output(outputs[output_index], manifest)
            generated = _resize_rgb(generated, (roi.shape[1], roi.shape[0]))
            result = rgb.copy()
            local = result[y0:y1, x0:x1]
            local[roi_mask > 0] = generated[roi_mask > 0]
            result[y0:y1, x0:x1] = local
            return result, {
                "status": "ok", "model_id": manifest.get("model_id"), "revision": manifest.get("revision"),
                "adapter": adapter, "backend": session.get_providers()[0], "roi": [x0, y0, x1, y1],
                "input_size": list(size), "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        except Exception as error:
            return None, {
                "status": "inference_failed", "role": "image_inpainting",
                "error": f"{type(error).__name__}: {error}",
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }

    def upscale_rgb(
        self,
        rgb: np.ndarray,
        mode: str,
        scale: int,
        cancel_check: Any | None = None,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Chạy SR ONNX theo tile; thiếu pack luôn báo lỗi, tuyệt đối không giả làm AI."""
        started = time.perf_counter()
        normalized_mode = str(mode).upper()
        if normalized_mode not in {"FAITHFUL", "SHARP"} or scale not in {2, 3, 4}:
            return None, {"status": "invalid_upscale_request"}
        native_scale = 4 if normalized_mode == "SHARP" and scale == 3 else scale
        role = f"upscale_{normalized_mode.lower()}_x{native_scale}"
        manifest = self._ready(role)
        if manifest is None:
            return None, {"status": "model_not_installed", "role": role}
        providers = self._providers(manifest)
        if ort is None or not providers:
            return None, {"status": "runtime_or_qualified_backend_unavailable", "role": role}
        try:
            adapter = str(manifest.get("adapter", "")).lower()
            if adapter not in {"swinir-v1", "realesrgan-v1", "super-resolution-v1", "generic-sr"}:
                return None, {"status": "unsupported_upscale_adapter", "adapter": adapter}
            if int(manifest.get("native_scale", native_scale)) != native_scale:
                return None, {"status": "native_scale_mismatch", "role": role}
            session = self._session(manifest)
            model_inputs = list(session.get_inputs())
            input_name = self._input_name(manifest, model_inputs, "input_name")
            layout = _layout_from_input(manifest, model_inputs[0], channels=3)
            backend = session.get_providers()[0]
            is_gpu = backend != "CPUExecutionProvider"
            tile_size = int(manifest.get("tile_size_gpu" if is_gpu else "tile_size_cpu", 256 if is_gpu else 128))
            overlap = int(manifest.get("tile_overlap_gpu" if is_gpu else "tile_overlap_cpu", 32 if is_gpu else 16))
            tile_size = max(16, min(tile_size, max(rgb.shape[:2])))
            overlap = max(0, min(overlap, tile_size // 2))
            output = np.zeros((rgb.shape[0] * native_scale, rgb.shape[1] * native_scale, 3), dtype=np.uint8)
            tile_count = 0
            for y in range(0, rgb.shape[0], tile_size):
                for x in range(0, rgb.shape[1], tile_size):
                    if cancel_check and cancel_check():
                        raise InterruptedError("Đã hủy upscale trước khi ghi file")
                    x1, y1 = min(rgb.shape[1], x + tile_size), min(rgb.shape[0], y + tile_size)
                    left, top = max(0, x - overlap), max(0, y - overlap)
                    right, bottom = min(rgb.shape[1], x1 + overlap), min(rgb.shape[0], y1 + overlap)
                    patch = rgb[top:bottom, left:right].astype(np.float32) / 255.0
                    if str(manifest.get("color_order", "RGB")).upper() == "BGR":
                        patch = patch[:, :, ::-1]
                    tensor = _tensor(patch, layout)
                    output_name = manifest.get("output_name")
                    outputs = session.run([str(output_name)] if output_name else None, {input_name: tensor})
                    output_index = int(manifest.get("output_index", 0))
                    if output_index < 0 or output_index >= len(outputs):
                        raise ValueError(f"Output index SR không hợp lệ: {output_index}")
                    enhanced = self._image_output(outputs[output_index], manifest)
                    expected = ((bottom - top) * native_scale, (right - left) * native_scale)
                    if enhanced.shape[:2] != expected:
                        raise ValueError(
                            f"Model SR trả {enhanced.shape[:2]}, cần {expected}; manifest/native_scale sai"
                        )
                    crop_top, crop_left = (y - top) * native_scale, (x - left) * native_scale
                    crop_bottom, crop_right = crop_top + (y1 - y) * native_scale, crop_left + (x1 - x) * native_scale
                    output[y * native_scale : y1 * native_scale, x * native_scale : x1 * native_scale] = enhanced[
                        crop_top:crop_bottom, crop_left:crop_right
                    ]
                    tile_count += 1
            if native_scale != scale:
                output = _resize_rgb(output, (rgb.shape[1] * scale, rgb.shape[0] * scale))
            return output, {
                "status": "ok", "model_id": manifest.get("model_id"), "revision": manifest.get("revision"),
                "adapter": adapter, "backend": backend, "native_scale": native_scale,
                "tile_size": tile_size, "tile_overlap": overlap, "tile_count": tile_count,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        except InterruptedError:
            return None, {"status": "cancelled", "role": role}
        except Exception as error:
            return None, {
                "status": "inference_failed", "role": role,
                "error": f"{type(error).__name__}: {error}",
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }

    def semantic_proposal(
        self,
        rgb: np.ndarray,
        foreground_points: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        started = time.perf_counter()
        manifest = self._ready("base_alpha_proposal")
        if manifest is None:
            return None, {"status": "model_not_installed"}
        providers = self._providers(manifest)
        if ort is None or not providers:
            return None, {"status": "runtime_or_qualified_backend_unavailable"}
        try:
            adapter = str(manifest.get("adapter", "generic")).lower()
            if adapter not in {"birefnet-v1", "birefnet", "generic", "onnx-alpha"}:
                return None, {"status": "unsupported_base_adapter", "adapter": adapter}
            session = self._session(manifest)
            model_inputs = list(session.get_inputs())
            input_name = self._input_name(manifest, model_inputs, "input_name")
            input_size = _input_size(manifest, (1024, 1024))

            def infer(source: np.ndarray) -> np.ndarray:
                tensor, _, mapping = self._letterboxed_image_input(
                    source, manifest, model_inputs[0], input_size
                )
                mask = self._run_mask(session, {input_name: tensor}, manifest)
                x0, y0, width, height = mapping
                mask = mask[y0 : y0 + height, x0 : x0 + width]
                return _resize_float(mask, (source.shape[1], source.shape[0]))

            proposal = infer(rgb).copy()
            focus_tiles: list[tuple[int, int, int, int]] = []
            points = _prompt_points(foreground_points, rgb.shape[1], rgb.shape[0])
            # Lite 512 chỉ chạy tile khi có click, tránh tăng thời gian và false positive vô cớ.
            if points and max(input_size) <= 512 and max(rgb.shape[:2]) > max(input_size):
                for point in points:
                    for vertical_bias in (0.0, -0.18):
                        bounds = _focus_tile_bounds(
                            rgb.shape[1], rgb.shape[0], point, input_size, vertical_bias
                        )
                        if bounds in focus_tiles:
                            continue
                        focus_tiles.append(bounds)
                        if len(focus_tiles) >= 3:
                            break
                    if len(focus_tiles) >= 3:
                        break
                for x0, y0, x1, y1 in focus_tiles:
                    detail = infer(rgb[y0:y1, x0:x1])
                    proposal[y0:y1, x0:x1] = np.maximum(proposal[y0:y1, x0:x1], detail)
            return proposal, {
                "status": "ok",
                "model_id": manifest.get("model_id"),
                "revision": manifest.get("revision"),
                "adapter": adapter,
                "backend": session.get_providers()[0],
                "input_size": list(input_size),
                "letterboxed": True,
                "focus_tile_count": len(focus_tiles),
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        except Exception as error:
            return None, {
                "status": "inference_failed",
                "error": f"{type(error).__name__}: {error}",
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }

    def topology_proposal(
        self,
        rgb: np.ndarray,
        semantic_alpha: np.ndarray,
        foreground_points: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        background_points: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Chạy model membership tương thích SAM2 nếu đã cài ONNX export hợp lệ."""
        started = time.perf_counter()
        manifest = self._ready("conditional_topology")
        if manifest is None:
            return None, {"status": "model_not_installed"}
        providers = self._providers(manifest)
        if ort is None or not providers:
            return None, {"status": "runtime_or_qualified_backend_unavailable"}
        try:
            adapter = str(manifest.get("adapter", "")).lower()
            if adapter not in {
                "sam2-conditional-v1",
                "sam2-mask-prompt-v1",
                "sam2-point-prompt-v1",
            }:
                return None, {"status": "unsupported_topology_adapter", "adapter": adapter}
            session = self._session(manifest)
            model_inputs = list(session.get_inputs())
            image_name = self._input_name(manifest, model_inputs, "input_name")
            image_tensor, layout = self._image_input(rgb, manifest, model_inputs[0], (1024, 1024))
            inputs: dict[str, np.ndarray] = {image_name: image_tensor}
            prompt_name = manifest.get("prompt_input_name")
            if prompt_name:
                prompt_name = self._input_name(manifest, model_inputs, "prompt_input_name", index=1)
                prompt_size = _input_size(manifest, (1024, 1024))
                prompt = _resize_float(semantic_alpha, prompt_size)
                inputs[prompt_name] = _tensor(prompt, layout)
            point_name = manifest.get("point_input_name")
            foreground = _prompt_points(foreground_points, rgb.shape[1], rgb.shape[0])
            background = _prompt_points(background_points, rgb.shape[1], rgb.shape[0])
            if point_name:
                point_index = 2 if prompt_name else 1
                point_name = self._input_name(manifest, model_inputs, "point_input_name", point_index)
                labelled_points = [(point, 1.0) for point in foreground] + [
                    (point, 0.0) for point in background
                ]
                if not labelled_points:
                    # ONNX export đủ điều kiện phải hiểu sentinel này như không có prompt.
                    labelled_points = [((-1.0, -1.0), -1.0)]
                point_size = _input_size(manifest, (1024, 1024))
                coordinates = np.asarray([item[0] for item in labelled_points], dtype=np.float32)
                coordinate_space = str(manifest.get("point_coordinate_space", "model")).lower()
                if coordinate_space == "model":
                    coordinates[:, 0] *= (point_size[0] - 1) / max(1, rgb.shape[1] - 1)
                    coordinates[:, 1] *= (point_size[1] - 1) / max(1, rgb.shape[0] - 1)
                elif coordinate_space == "normalized":
                    coordinates[:, 0] /= max(1, rgb.shape[1] - 1)
                    coordinates[:, 1] /= max(1, rgb.shape[0] - 1)
                elif coordinate_space != "source":
                    raise ValueError(f"point_coordinate_space không hỗ trợ: {coordinate_space}")
                inputs[point_name] = coordinates[None, :, :]
                label_name = self._input_name(
                    manifest,
                    model_inputs,
                    "point_label_input_name",
                    point_index + 1,
                )
                inputs[label_name] = np.asarray(
                    [[item[1] for item in labelled_points]], dtype=np.float32
                )
            proposal = self._run_mask(session, inputs, manifest)
            proposal = _resize_float(proposal, (rgb.shape[1], rgb.shape[0]))
            return proposal, {
                "status": "ok",
                "model_id": manifest.get("model_id"),
                "revision": manifest.get("revision"),
                "adapter": adapter,
                "backend": session.get_providers()[0],
                "membership_only": True,
                "prompt_count": len(foreground) + len(background),
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
        sure_foreground: np.ndarray | None = None,
        sure_background: np.ndarray | None = None,
        unknown: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run ViTMatte only on a boundary ROI; clamp known regions afterwards."""
        started = time.perf_counter()
        manifest = self._ready("roi_matting")
        if manifest is None:
            return alpha, {"status": "model_not_installed"}
        providers = self._providers(manifest)
        if ort is None or not providers:
            return alpha, {"status": "runtime_or_qualified_backend_unavailable"}
        uses_external_trimap = any(
            value is not None for value in (sure_foreground, sure_background, unknown)
        )
        if uses_external_trimap:
            # Trimap từ object candidate là nguồn chân lý; không suy ngược từ alpha đã hỏng.
            known_foreground = np.asarray(
                sure_foreground if sure_foreground is not None else np.zeros(alpha.shape, dtype=bool),
                dtype=bool,
            )
            known_background = np.asarray(
                sure_background if sure_background is not None else np.zeros(alpha.shape, dtype=bool),
                dtype=bool,
            )
            unknown_mask = np.asarray(
                unknown if unknown is not None else ~(known_foreground | known_background),
                dtype=bool,
            )
            if (
                known_foreground.shape != alpha.shape
                or known_background.shape != alpha.shape
                or unknown_mask.shape != alpha.shape
            ):
                return alpha, {"status": "invalid_trimap_shape"}
            known_background &= ~known_foreground
            unknown_mask &= ~(known_foreground | known_background)
        else:
            foreground = alpha >= np.minimum(source_alpha, 0.95)
            background = alpha <= 0.02
            # Dải morphology giữ hành vi cũ cho V3/V1 khi chưa có candidate AI.
            fg_image = Image.fromarray(foreground.astype(np.uint8) * 255, "L")
            bg_image = Image.fromarray(background.astype(np.uint8) * 255, "L")
            known_foreground = np.asarray(
                fg_image.filter(ImageFilter.MinFilter(9)), dtype=np.uint8
            ) > 0
            known_background = np.asarray(
                bg_image.filter(ImageFilter.MinFilter(9)), dtype=np.uint8
            ) > 0
            unknown_mask = ~(known_foreground | known_background)
        unknown = unknown_mask
        ys, xs = np.nonzero(unknown)
        if not xs.size:
            return alpha, {"status": "no_unknown_roi"}
        margin = 32
        x0, x1 = max(0, int(xs.min()) - margin), min(rgb.shape[1], int(xs.max()) + margin + 1)
        y0, y1 = max(0, int(ys.min()) - margin), min(rgb.shape[0], int(ys.max()) + margin + 1)
        try:
            adapter = str(manifest.get("adapter", "vitmatte-v1")).lower()
            if adapter not in {"vitmatte-v1", "vitmatte", "generic"}:
                return alpha, {"status": "unsupported_matting_adapter", "adapter": adapter}
            session = self._session(manifest)
            model_inputs = list(session.get_inputs())
            input_name = self._input_name(manifest, model_inputs, "input_name")
            width, height = _input_size(manifest, (512, 512))
            result = alpha.copy()
            roi_width, roi_height = x1 - x0, y1 - y0
            # ViTMatte chỉ thấy tile 512 thay vì toàn bộ quai/cốc bị nén về 512 px.
            configured_tile = int(manifest.get("tile_size", min(width, height)))
            tile_width = max(1, min(configured_tile, roi_width))
            tile_height = max(1, min(configured_tile, roi_height))
            configured_overlap = int(manifest.get("tile_overlap", 96))
            overlap_x = min(max(0, configured_overlap), max(0, tile_width - 1))
            overlap_y = min(max(0, configured_overlap), max(0, tile_height - 1))

            def starts(length: int, tile: int, overlap: int) -> list[int]:
                if length <= tile:
                    return [0]
                stride = max(1, tile - overlap)
                values = list(range(0, max(1, length - tile + 1), stride))
                last = length - tile
                if values[-1] != last:
                    values.append(last)
                return values

            matte_sum = np.zeros((roi_height, roi_width), dtype=np.float32)
            weight_sum = np.zeros((roi_height, roi_width), dtype=np.float32)
            tile_count = 0
            for local_y in starts(roi_height, tile_height, overlap_y):
                for local_x in starts(roi_width, tile_width, overlap_x):
                    local_x1 = local_x + tile_width
                    local_y1 = local_y + tile_height
                    crop_rgb = rgb[y0 + local_y : y0 + local_y1, x0 + local_x : x0 + local_x1]
                    image_tensor, layout = self._image_input(
                        crop_rgb, manifest, model_inputs[0], (width, height)
                    )
                    trimap = np.full((tile_height, tile_width), 0.5, dtype=np.float32)
                    local_background = known_background[
                        y0 + local_y : y0 + local_y1, x0 + local_x : x0 + local_x1
                    ]
                    local_foreground = known_foreground[
                        y0 + local_y : y0 + local_y1, x0 + local_x : x0 + local_x1
                    ]
                    trimap[local_background] = 0.0
                    trimap[local_foreground] = 1.0
                    trimap_tensor = _tensor(_resize_float(trimap, (width, height)), layout)
                    trimap_name = manifest.get("trimap_input_name")
                    if trimap_name:
                        trimap_name = self._input_name(
                            manifest, model_inputs, "trimap_input_name", index=1
                        )
                        inputs: dict[str, np.ndarray] = {
                            input_name: image_tensor,
                            trimap_name: trimap_tensor,
                        }
                    elif layout == "NHWC":
                        inputs = {input_name: np.concatenate((image_tensor, trimap_tensor), axis=-1)}
                    else:
                        inputs = {input_name: np.concatenate((image_tensor, trimap_tensor), axis=1)}
                    matte = _resize_float(
                        self._run_mask(session, inputs, manifest), (tile_width, tile_height)
                    )
                    weights = _hann_weight(tile_height, tile_width)
                    matte_sum[local_y:local_y1, local_x:local_x1] += matte * weights
                    weight_sum[local_y:local_y1, local_x:local_x1] += weights
                    tile_count += 1

            matte = matte_sum / np.maximum(weight_sum, 1e-6)
            local_unknown = unknown[y0:y1, x0:x1]
            view = result[y0:y1, x0:x1]
            view[local_unknown] = matte[local_unknown] * source_alpha[y0:y1, x0:x1][local_unknown]
            result[known_background] = 0.0
            result[known_foreground] = source_alpha[known_foreground]
            return np.minimum(np.clip(result, 0.0, 1.0), source_alpha), {
                "status": "ok",
                "model_id": manifest.get("model_id"),
                "revision": manifest.get("revision"),
                "adapter": adapter,
                "backend": session.get_providers()[0],
                "roi": [x0, y0, x1, y1],
                "tile_count": tile_count,
                "tile_size": [tile_width, tile_height],
                "tile_overlap": [overlap_x, overlap_y],
                "external_trimap": uses_external_trimap,
                "unknown_fraction": round(float(np.mean(unknown)), 6),
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        except Exception as error:
            return alpha, {
                "status": "inference_failed",
                "error": f"{type(error).__name__}: {error}",
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
