from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .mask import bounds_from_mask, hard_mask


def restoration_weight(soft_mask: np.ndarray) -> np.ndarray:
    """Lấp dứt điểm lõi mask và chỉ feather ở vành ngoài để tránh watermark còn sót."""
    values = np.clip(np.asarray(soft_mask, dtype=np.float32), 0.0, 1.0)
    return np.clip((values - 0.01) / 0.34, 0.0, 1.0).astype(np.float32)


def expanded_bounds(
    mask: np.ndarray,
    width: int,
    height: int,
    quality: str = "BALANCED",
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bounds_from_mask(mask, 0.01)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0)
    extent = max(x1 - x0, y1 - y0)
    multiplier = 2.0 if str(quality).upper() == "FAST" else 3.0
    if str(quality).upper() == "MAXIMUM":
        multiplier = 4.0
    margin = max(12, min(512, int(round(extent * multiplier))))
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(width, x1 + margin),
        min(height, y1 + margin),
    )


def restore_roi_with_candidate(
    rgb: np.ndarray,
    soft_mask: np.ndarray,
    candidate_roi: np.ndarray,
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = roi
    output = rgb.copy()
    local_soft = restoration_weight(soft_mask[y0:y1, x0:x1])[:, :, None]
    original = rgb[y0:y1, x0:x1].astype(np.float32)
    candidate = candidate_roi.astype(np.float32)
    blended = original * (1.0 - local_soft) + candidate * local_soft
    changed = local_soft[:, :, 0] > 0.005
    view = output[y0:y1, x0:x1]
    view[changed] = np.rint(np.clip(blended[changed], 0, 255)).astype(np.uint8)
    return output


def telea_restore(rgb: np.ndarray, soft_mask: np.ndarray, quality: str = "BALANCED") -> tuple[np.ndarray, dict[str, Any]]:
    height, width = soft_mask.shape
    roi = expanded_bounds(soft_mask, width, height, quality)
    if roi == (0, 0, 0, 0):
        raise ValueError("Chưa có vùng watermark để xóa")
    x0, y0, x1, y1 = roi
    local_rgb = rgb[y0:y1, x0:x1]
    local_mask = hard_mask(soft_mask[y0:y1, x0:x1])
    radius = 3 if str(quality).upper() == "FAST" else 4
    if str(quality).upper() == "MAXIMUM":
        radius = 5
    repaired_roi = cv2.inpaint(local_rgb, local_mask, radius, cv2.INPAINT_TELEA)
    return restore_roi_with_candidate(rgb, soft_mask, repaired_roi, roi), {
        "route": "TELEA_FAST",
        "roi": list(roi),
        "radius": radius,
    }


def ai_restore(
    rgb: np.ndarray,
    soft_mask: np.ndarray,
    runtime: Any,
    quality: str,
    role: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if runtime is None or not hasattr(runtime, "inpaint_rgb"):
        return None, {"status": "runtime_unavailable", "role": role}
    height, width = soft_mask.shape
    roi = expanded_bounds(soft_mask, width, height, quality)
    if roi == (0, 0, 0, 0):
        return None, {"status": "empty_mask", "role": role}
    x0, y0, x1, y1 = roi
    repaired_roi, diagnostics = runtime.inpaint_rgb(
        rgb[y0:y1, x0:x1],
        soft_mask[y0:y1, x0:x1],
        role=role,
        context={"quality": str(quality).upper(), "roi": list(roi)},
    )
    if repaired_roi is None:
        return None, diagnostics
    return restore_roi_with_candidate(rgb, soft_mask, repaired_roi, roi), {
        **diagnostics,
        "route": "AI_QUALITY" if role.endswith("quality") else "AI_FAST",
        "roi": list(roi),
    }
