from __future__ import annotations

import cv2
import numpy as np


def attenuate_transparent_overlay(rgb: np.ndarray, soft_mask: np.ndarray) -> np.ndarray:
    """Giảm residual lớp phủ mờ trong mask mà vẫn giữ nhiều detail gốc nhất có thể."""
    values = np.clip(np.asarray(soft_mask, dtype=np.float32), 0.0, 1.0)
    if not np.any(values > 0.01):
        return rgb.copy()
    smooth = cv2.bilateralFilter(rgb, d=0, sigmaColor=24, sigmaSpace=9).astype(np.float32)
    source = rgb.astype(np.float32)
    residual = source - smooth
    strength = (values[:, :, None] * 0.72).astype(np.float32)
    restored = source - residual * strength
    output = source.copy()
    output[values > 0.01] = restored[values > 0.01]
    return np.rint(np.clip(output, 0, 255)).astype(np.uint8)
