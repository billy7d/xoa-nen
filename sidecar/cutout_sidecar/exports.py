from __future__ import annotations

import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageCms

from .model_runtime import LocalModelRuntime


OUTPUT_MODES = {"MASTER_SOURCE_FAITHFUL", "POD_READY", "ALPHA_ONLY"}


def _atomic_save(
    image: Image.Image,
    destination: Path,
    cancel_check: Any | None = None,
    **save_options: Any,
) -> None:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if cancel_check and cancel_check():
            raise InterruptedError("Đã hủy trước khi ghi file")
        image.save(temporary, format="PNG", **save_options)
        # Windows chỉ cho fsync ổn định với handle có quyền ghi.
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        if cancel_check and cancel_check():
            raise InterruptedError("Đã hủy trước khi hoàn tất file")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _srgb_profile_bytes() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _convert_to_srgb(rgb: np.ndarray, icc_profile: bytes | None) -> np.ndarray:
    image = Image.fromarray(rgb, "RGB")
    if not icc_profile:
        return rgb.copy()
    try:
        source = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
        converted = ImageCms.profileToProfile(
            image,
            source,
            ImageCms.createProfile("sRGB"),
            outputMode="RGB",
        )
        return np.asarray(converted, dtype=np.uint8).copy()
    except (ImageCms.PyCMSError, OSError, ValueError):
        # ICC lỗi không được làm hỏng preview hoặc thao tác xóa watermark.
        return rgb.copy()


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32) / 255.0
    return np.where(values <= 0.04045, values / 12.92, np.power((values + 0.055) / 1.055, 2.4))


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * np.power(values, 1.0 / 2.4) - 0.055)


def _background_linear(
    rgb: np.ndarray,
    fractional: np.ndarray,
    background_rgb: list[float] | list[list[float]] | dict[str, Any],
) -> np.ndarray:
    """Lấy màu nền cục bộ ở linear RGB cho từng pixel bán trong suốt."""
    observed = rgb[fractional].astype(np.float32)
    if isinstance(background_rgb, dict) and background_rgb.get("coefficients_linear"):
        ys, xs = np.nonzero(fractional)
        x = (xs.astype(np.float32) / max(1, rgb.shape[1] - 1)) * 2.0 - 1.0
        y = (ys.astype(np.float32) / max(1, rgb.shape[0] - 1)) * 2.0 - 1.0
        order = int(background_rgb.get("order", 0))
        if order == 0:
            design = np.ones((x.size, 1), dtype=np.float32)
        elif order == 1:
            design = np.stack((np.ones_like(x), x, y), axis=1)
        else:
            design = np.stack((np.ones_like(x), x, y, x * y, x * x, y * y), axis=1)
        return np.clip(design @ np.asarray(background_rgb["coefficients_linear"], dtype=np.float32), 0.0, 1.0)
    palette = np.asarray(background_rgb, dtype=np.float32).reshape(-1, 3)
    distances = np.sum(np.square(observed[:, None, :] - palette[None, :, :]), axis=2)
    return _srgb_to_linear(palette[np.argmin(distances, axis=1)])


def _decontaminate_edges(
    rgb: np.ndarray,
    alpha: np.ndarray,
    background_rgb: list[float] | list[list[float]] | dict[str, Any] | None,
) -> np.ndarray:
    if not background_rgb:
        return rgb
    result = rgb.astype(np.float32)
    fractional = (alpha > 0.005) & (alpha < 0.995)
    if not np.any(fractional):
        return rgb
    observed_linear = _srgb_to_linear(result[fractional])
    background_linear = _background_linear(rgb, fractional, background_rgb)
    fractional_alpha = np.maximum(alpha[fractional, None], 0.005)
    estimated = np.clip(
        (observed_linear - (1.0 - fractional_alpha) * background_linear) / fractional_alpha,
        0.0,
        1.0,
    )
    # Chỉ sửa dải alpha mềm; vùng đặc gần như không đổi để bảo vệ màu artwork/logo.
    blend = np.clip((0.92 - alpha[fractional, None]) / 0.90, 0.0, 1.0) * 0.90
    clean_linear = observed_linear * (1.0 - blend) + estimated * blend
    result[fractional] = _linear_to_srgb(clean_linear) * 255.0
    return np.rint(np.clip(result, 0.0, 255.0)).astype(np.uint8)


def pod_clean_rgb(
    rgb: np.ndarray,
    alpha: np.ndarray,
    icc_profile: bytes | None,
    background_rgb: list[float] | list[list[float]] | dict[str, Any] | None,
) -> np.ndarray:
    """Bản RGB chỉ cho POD/preview; không bao giờ ghi ngược vào Master canonical."""
    return _decontaminate_edges(_convert_to_srgb(rgb, icc_profile), alpha, background_rgb)


def _resize_alpha_with_support(alpha: np.ndarray, scale: int) -> np.ndarray:
    """Nâng alpha 16-bit riêng, giữ lỗ quai và không dùng model sinh ảnh cho alpha."""
    width, height = alpha.shape[1] * scale, alpha.shape[0] * scale
    resized = np.asarray(
        Image.fromarray(np.asarray(alpha, dtype=np.float32), "F").resize(
            (width, height), Image.Resampling.LANCZOS
        ),
        dtype=np.float32,
    )
    support = np.asarray(
        Image.fromarray((alpha > 0.001).astype(np.uint8) * 255, "L").resize(
            (width, height), Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    ) > 0
    return np.where(support, np.clip(resized, 0.0, 1.0), 0.0).astype(np.float32)


def _trim_and_pad(
    rgb: np.ndarray,
    alpha: np.ndarray,
    trim: bool,
    padding: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not trim:
        return rgb, alpha
    ys, xs = np.nonzero(alpha > 0.001)
    if not xs.size:
        return rgb, alpha
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    cropped_rgb = rgb[y0:y1, x0:x1]
    cropped_alpha = alpha[y0:y1, x0:x1]
    padding = max(0, int(padding))
    if padding == 0:
        return cropped_rgb, cropped_alpha
    out_rgb = np.zeros(
        (cropped_rgb.shape[0] + padding * 2, cropped_rgb.shape[1] + padding * 2, 3),
        dtype=np.uint8,
    )
    out_alpha = np.zeros(out_rgb.shape[:2], dtype=np.float32)
    out_rgb[padding:-padding, padding:-padding] = cropped_rgb
    out_alpha[padding:-padding, padding:-padding] = cropped_alpha
    return out_rgb, out_alpha


def _place_on_canvas(
    rgb: np.ndarray,
    alpha: np.ndarray,
    canvas: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not canvas:
        return rgb, alpha
    canvas_width = int(canvas.get("width", rgb.shape[1]))
    canvas_height = int(canvas.get("height", rgb.shape[0]))
    if canvas_width < rgb.shape[1] or canvas_height < rgb.shape[0]:
        raise ValueError("Canvas nhỏ hơn artwork; ứng dụng không tự scale hoặc clip")
    out_rgb = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    out_alpha = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    x = (canvas_width - rgb.shape[1]) // 2
    y = (canvas_height - rgb.shape[0]) // 2
    out_rgb[y : y + rgb.shape[0], x : x + rgb.shape[1]] = rgb
    out_alpha[y : y + alpha.shape[0], x : x + alpha.shape[1]] = alpha
    return out_rgb, out_alpha


def export_image(
    output_mode: str,
    destination: str | Path,
    rgb: np.ndarray,
    alpha: np.ndarray,
    icc_profile: bytes | None,
    background_rgb: list[float] | list[list[float]] | dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    runtime: LocalModelRuntime | None = None,
    cancel_check: Any | None = None,
) -> dict[str, Any]:
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"Output mode không hợp lệ: {output_mode}")
    settings = settings or {}
    destination_path = Path(destination)
    ppi = float(settings.get("target_ppi") or 300.0)
    upscale_mode = str(settings.get("upscale_mode", "NONE")).upper()
    upscale_scale = int(settings.get("upscale_scale", 1))
    if upscale_mode not in {"NONE", "FAITHFUL", "SHARP"}:
        raise ValueError("upscale_mode phải là NONE, FAITHFUL hoặc SHARP")
    if upscale_scale not in {1, 2, 3, 4}:
        raise ValueError("upscale_scale phải là 1, 2, 3 hoặc 4")
    if upscale_mode == "NONE" and upscale_scale != 1:
        raise ValueError("Scale x2/x3/x4 cần chọn chế độ AI FAITHFUL hoặc SHARP")
    if upscale_mode != "NONE" and upscale_scale == 1:
        raise ValueError("Chế độ AI cần scale x2, x3 hoặc x4")
    if upscale_mode != "NONE" and output_mode != "POD_READY":
        raise ValueError("Upscale chỉ áp dụng cho POD_READY; Master và Alpha-only giữ native")

    if output_mode == "ALPHA_ONLY":
        alpha16 = np.rint(np.clip(alpha, 0.0, 1.0) * 65535.0).astype("<u2")
        image = Image.fromarray(alpha16)
        _atomic_save(image, destination_path, cancel_check=cancel_check, compress_level=6, dpi=(ppi, ppi))
        return {
            "path": str(destination_path.resolve()),
            "mode": output_mode,
            "width": alpha.shape[1],
            "height": alpha.shape[0],
            "bit_depth": 16,
        }

    export_rgb = rgb.copy()
    export_alpha = alpha.copy()
    output_icc = icc_profile

    if output_mode == "POD_READY":
        export_rgb = pod_clean_rgb(export_rgb, export_alpha, icc_profile, background_rgb)
        output_icc = _srgb_profile_bytes()
        export_rgb, export_alpha = _trim_and_pad(
            export_rgb,
            export_alpha,
            bool(settings.get("trim", False)),
            int(settings.get("padding", 0)),
        )
        export_rgb, export_alpha = _place_on_canvas(
            export_rgb, export_alpha, settings.get("canvas")
        )

        if upscale_mode != "NONE":
            if cancel_check and cancel_check():
                raise InterruptedError("Đã hủy trước khi chạy model upscale")
            if runtime is None:
                raise RuntimeError(
                    "Chưa có runtime model upscale đã qualification; app không dùng Lanczos giả AI."
                )
            native_width, native_height = export_rgb.shape[1], export_rgb.shape[0]
            enhanced_rgb, diagnostics = runtime.upscale_rgb(
                export_rgb, upscale_mode, upscale_scale, cancel_check=cancel_check
            )
            if enhanced_rgb is None:
                status = diagnostics.get("status")
                if status == "cancelled":
                    raise InterruptedError("Đã hủy export upscale")
                raise RuntimeError(
                    "Không thể chạy AI upscale: "
                    f"{status}. Hãy cài model-pack ONNX đã ký/qualification; app không dùng Lanczos giả AI."
                )
            export_rgb = enhanced_rgb
            export_alpha = _resize_alpha_with_support(export_alpha, upscale_scale)
            upscale_details = {
                **diagnostics,
                "mode": upscale_mode,
                "scale": upscale_scale,
                "native_width": native_width,
                "native_height": native_height,
            }
        else:
            upscale_details = None
    else:
        upscale_details = None

    alpha8 = np.rint(np.clip(export_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba = np.dstack((export_rgb, alpha8))
    image = Image.fromarray(rgba, "RGBA")
    options: dict[str, Any] = {"compress_level": 6, "dpi": (ppi, ppi)}
    if output_icc:
        options["icc_profile"] = output_icc
    _atomic_save(image, destination_path, cancel_check=cancel_check, **options)
    result = {
        "path": str(destination_path.resolve()),
        "mode": output_mode,
        "width": image.width,
        "height": image.height,
        "bit_depth": 8,
        "straight_alpha": True,
        "rgb_identical": output_mode == "MASTER_SOURCE_FAITHFUL",
    }
    if upscale_details:
        result["upscale"] = upscale_details
        result["model"] = upscale_details.get("model_id")
        result["backend"] = upscale_details.get("backend")
        result["native_size"] = [upscale_details["native_width"], upscale_details["native_height"]]
        result["output_size"] = [image.width, image.height]
        result["latency_ms"] = upscale_details.get("latency_ms")
        if upscale_mode == "SHARP":
            result["warnings"] = ["SHARP có thể thay đổi chữ/logo nhỏ; ưu tiên FAITHFUL cho artwork POD."]
    return result
