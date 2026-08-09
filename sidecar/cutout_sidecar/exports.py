from __future__ import annotations

import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageCms


OUTPUT_MODES = {"MASTER_SOURCE_FAITHFUL", "POD_READY", "ALPHA_ONLY"}


def _atomic_save(image: Image.Image, destination: Path, **save_options: Any) -> None:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format="PNG", **save_options)
        # Windows chỉ cho fsync ổn định với handle có quyền ghi.
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
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
        return rgb.copy()


def _decontaminate_edges(
    rgb: np.ndarray,
    alpha: np.ndarray,
    background_rgb: list[float] | list[list[float]] | dict[str, Any] | None,
) -> np.ndarray:
    if not background_rgb:
        return rgb
    result = rgb.astype(np.float32)
    fractional = (alpha > 0.03) & (alpha < 0.98)
    if not np.any(fractional):
        return rgb
    observed = result[fractional]
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
        coefficients = np.asarray(background_rgb["coefficients_linear"], dtype=np.float32)
        linear = np.clip(design @ coefficients, 0.0, 1.0)
        srgb = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
        )
        background = np.clip(srgb * 255.0, 0.0, 255.0)
    else:
        palette = np.asarray(background_rgb, dtype=np.float32).reshape(-1, 3)
        distances = np.sum(np.square(observed[:, None, :] - palette[None, :, :]), axis=2)
        background = palette[np.argmin(distances, axis=1)]
    fractional_alpha = np.maximum(alpha[fractional, None], 0.03)
    estimated = (observed - (1.0 - fractional_alpha) * background) / fractional_alpha
    estimated = np.clip(estimated, 0.0, 255.0)
    blend = np.clip((0.98 - alpha[fractional, None]) / 0.95, 0.0, 1.0) * 0.85
    result[fractional] = observed * (1.0 - blend) + estimated * blend
    return np.rint(np.clip(result, 0.0, 255.0)).astype(np.uint8)


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
) -> dict[str, Any]:
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"Output mode không hợp lệ: {output_mode}")
    settings = settings or {}
    destination_path = Path(destination)
    ppi = float(settings.get("target_ppi") or 300.0)

    if output_mode == "ALPHA_ONLY":
        alpha16 = np.rint(np.clip(alpha, 0.0, 1.0) * 65535.0).astype("<u2")
        image = Image.fromarray(alpha16)
        _atomic_save(image, destination_path, compress_level=6, dpi=(ppi, ppi))
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
        export_rgb = _convert_to_srgb(export_rgb, icc_profile)
        export_rgb = _decontaminate_edges(export_rgb, export_alpha, background_rgb)
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

    alpha8 = np.rint(np.clip(export_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba = np.dstack((export_rgb, alpha8))
    image = Image.fromarray(rgba, "RGBA")
    options: dict[str, Any] = {"compress_level": 6, "dpi": (ppi, ppi)}
    if output_icc:
        options["icc_profile"] = output_icc
    _atomic_save(image, destination_path, **options)
    return {
        "path": str(destination_path.resolve()),
        "mode": output_mode,
        "width": image.width,
        "height": image.height,
        "bit_depth": 8,
        "straight_alpha": True,
        "rgb_identical": output_mode == "MASTER_SOURCE_FAITHFUL",
    }
