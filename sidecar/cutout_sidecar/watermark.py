"""Facade tương thích cho Watermark Removal v2.

Logic chính nằm trong ``watermark_engine`` để tách detector, mask session và
restoration router khỏi coordinator.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .watermark_engine import (
    bounds_from_mask,
    confidence_to_soft_mask,
    detect_watermark,
    hard_mask,
    rasterize_stroke,
    restore_watermark,
)


def automatic_watermark_mask(
    rgb: np.ndarray,
    feather: float = 8.0,
    expand: str = "MEDIUM",
) -> tuple[np.ndarray, dict[str, Any]]:
    detection = detect_watermark(rgb)
    soft = confidence_to_soft_mask(detection.confidence, feather=feather, expand=expand)
    mask = hard_mask(soft)
    diagnostics = dict(detection.diagnostics)
    diagnostics["soft_pixels"] = int(np.count_nonzero(soft > 0.01))
    diagnostics["bounds"] = list(bounds_from_mask(soft, 0.01))
    return mask, diagnostics


def brush_mask(
    shape: tuple[int, int],
    points: list[dict[str, float]],
    radius: float,
    hardness: float = 1.0,
    feather: float = 0.0,
) -> np.ndarray:
    soft, _ = rasterize_stroke(shape, points, radius, hardness=hardness, feather=feather)
    return hard_mask(soft)


def inpaint_watermark(
    rgb: np.ndarray,
    mask: np.ndarray,
    quality: str = "BALANCED",
    runtime: Any | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    soft = np.clip(np.asarray(mask, dtype=np.float32) / 255.0, 0.0, 1.0)
    repaired, bounds, _diagnostics = restore_watermark(rgb, soft, quality=quality, runtime=runtime)
    return repaired, bounds
