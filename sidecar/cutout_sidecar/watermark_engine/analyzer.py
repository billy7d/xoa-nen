from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .mask import bounds_from_mask


def analyze_watermark(rgb: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(mask, dtype=np.float32)
    selected = values > 0.08
    height, width = values.shape
    if not np.any(selected):
        return {
            "mask_ratio": 0.0,
            "texture": "EMPTY",
            "structure_score": 0.0,
            "transparency_score": 0.0,
            "semantic_complexity": 0.0,
            "bounds": [0, 0, 0, 0],
        }
    x0, y0, x1, y1 = bounds_from_mask(selected, 0)
    margin = max(8, min(96, round(max(x1 - x0, y1 - y0) * 0.35)))
    rx0 = max(0, x0 - margin)
    ry0 = max(0, y0 - margin)
    rx1 = min(width, x1 + margin)
    ry1 = min(height, y1 + margin)
    roi = rgb[ry0:ry1, rx0:rx1]
    roi_mask = selected[ry0:ry1, rx0:rx1]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    gradients_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradients_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = cv2.magnitude(gradients_x, gradients_y)
    edge_threshold = max(8.0, float(np.percentile(gradient_mag, 75)))
    edge_mask = gradient_mag >= edge_threshold
    structure_score = float(np.mean(edge_mask[roi_mask])) if np.any(roi_mask) else 0.0
    context = ~roi_mask
    texture_energy = float(np.std(gray[context])) if np.any(context) else float(np.std(gray))
    high_frequency = float(np.mean(gradient_mag[context])) if np.any(context) else float(np.mean(gradient_mag))
    mask_ratio = float(np.count_nonzero(selected) / max(1, height * width))
    if texture_energy < 7.0 and high_frequency < 6.0:
        texture = "FLAT"
    elif texture_energy < 18.0 and high_frequency < 14.0:
        texture = "GRADIENT"
    elif structure_score > 0.26:
        texture = "GEOMETRIC"
    elif texture_energy > 38.0 or high_frequency > 28.0:
        texture = "COMPLEX"
    else:
        texture = "TEXTURE"

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(3.0, min(width, height) / 120.0))
    residual = np.abs(gray.astype(np.float32) - blur.astype(np.float32))
    inside_residual = float(np.mean(residual[roi_mask])) if np.any(roi_mask) else 0.0
    outside_residual = float(np.mean(residual[context])) if np.any(context) else inside_residual
    transparency_score = float(np.clip(1.0 - inside_residual / max(12.0, outside_residual * 2.4), 0.0, 1.0))
    semantic_complexity = float(np.clip((texture_energy / 64.0) * 0.55 + structure_score * 0.45, 0.0, 1.0))
    return {
        "mask_ratio": round(mask_ratio, 6),
        "texture": texture,
        "structure_score": round(structure_score, 6),
        "transparency_score": round(transparency_score, 6),
        "semantic_complexity": round(semantic_complexity, 6),
        "bounds": [x0, y0, x1, y1],
        "roi": [rx0, ry0, rx1, ry1],
        "texture_energy": round(texture_energy, 6),
        "high_frequency": round(high_frequency, 6),
    }
