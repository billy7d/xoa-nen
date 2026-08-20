from __future__ import annotations

import cv2
import numpy as np


def harmonize_boundary(original: np.ndarray, candidate: np.ndarray, soft_mask: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(soft_mask, dtype=np.float32), 0.0, 1.0)
    core = values > 0.08
    if not np.any(core):
        return candidate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    outer = cv2.dilate(core.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = outer & ~core
    if not np.any(ring):
        return candidate
    result = candidate.astype(np.float32).copy()
    original_f = original.astype(np.float32)
    candidate_f = candidate.astype(np.float32)
    delta = np.mean(original_f[ring] - candidate_f[ring], axis=0)
    inner_weight = np.clip(values[:, :, None] * 0.18, 0.0, 0.18)
    result[core] = np.clip(result[core] + delta * inner_weight[core], 0, 255)
    return np.rint(result).astype(np.uint8)
