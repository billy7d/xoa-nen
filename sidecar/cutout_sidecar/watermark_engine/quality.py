from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def score_candidate(original: np.ndarray, candidate: np.ndarray, soft_mask: np.ndarray) -> dict[str, Any]:
    values = np.clip(np.asarray(soft_mask, dtype=np.float32), 0.0, 1.0)
    core = values > 0.08
    if not np.any(core):
        return {"overall": 0.0, "seam": 0.0, "texture": 0.0, "edge": 0.0, "residual_watermark": 1.0}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    outer = cv2.dilate(core.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = outer & ~core
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
    diff = np.abs(original.astype(np.float32) - candidate.astype(np.float32))
    change_inside = float(np.mean(diff[core])) if np.any(core) else 0.0
    seam_delta = float(np.mean(np.abs(original_gray[ring].astype(np.float32) - candidate_gray[ring].astype(np.float32)))) if np.any(ring) else 0.0
    original_edges = cv2.Sobel(original_gray, cv2.CV_32F, 1, 1, ksize=3)
    candidate_edges = cv2.Sobel(candidate_gray, cv2.CV_32F, 1, 1, ksize=3)
    edge_delta = float(np.mean(np.abs(original_edges[ring] - candidate_edges[ring]))) if np.any(ring) else 0.0
    seam = float(np.clip(1.0 - seam_delta / 30.0, 0.0, 1.0))
    edge = float(np.clip(1.0 - edge_delta / 80.0, 0.0, 1.0))
    texture = float(np.clip(1.0 - abs(float(np.std(candidate_gray[core])) - float(np.std(original_gray[outer]))) / 42.0, 0.0, 1.0))
    residual_watermark = float(np.clip(1.0 - change_inside / 42.0, 0.0, 1.0))
    overall = seam * 0.38 + texture * 0.27 + edge * 0.22 + (1.0 - residual_watermark) * 0.13
    return {
        "overall": round(float(overall), 6),
        "seam": round(seam, 6),
        "texture": round(texture, 6),
        "edge": round(edge, 6),
        "residual_watermark": round(residual_watermark, 6),
        "changed_mean": round(change_inside, 6),
    }
