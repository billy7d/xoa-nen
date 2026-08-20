from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .mask import bounds_from_mask


@dataclass(frozen=True)
class WatermarkDetection:
    confidence: np.ndarray
    diagnostics: dict[str, Any]


def _normalise(values: np.ndarray, percentile: float = 96.0) -> np.ndarray:
    scale = float(np.percentile(values, percentile))
    if not np.isfinite(scale) or scale <= 1e-6:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip(values.astype(np.float32) / scale, 0.0, 1.0)


def _local_residual_score(channel: np.ndarray, sigma: float) -> np.ndarray:
    blur = cv2.GaussianBlur(channel, (0, 0), sigmaX=sigma)
    residual = np.abs(channel.astype(np.float32) - blur.astype(np.float32))
    # Mean absolute residual là xấp xỉ robust đủ nhanh cho ảnh lớn, tránh threshold global.
    local_mad = cv2.GaussianBlur(residual, (0, 0), sigmaX=max(1.5, sigma * 2.4)) + 3.0
    return np.clip(residual / (local_mad * 2.15), 0.0, 1.0)


def _multi_scale_residual(gray: np.ndarray) -> np.ndarray:
    scores = [_local_residual_score(gray, sigma) for sigma in (1.5, 3.0, 6.0, 12.0)]
    return np.maximum.reduce(scores).astype(np.float32)


def _color_overlay_score(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    channels: list[np.ndarray] = []
    for channel in (rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2], lab[:, :, 1], lab[:, :, 2]):
        channels.append(_multi_scale_residual(channel))
    return np.maximum.reduce(channels).astype(np.float32)


def _edge_score(gray: np.ndarray) -> np.ndarray:
    median = float(np.median(gray))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median + 24))
    edges = cv2.Canny(gray, lower, upper)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return (edges.astype(np.float32) / 255.0)


def _mser_text_score(gray: np.ndarray) -> tuple[np.ndarray, int]:
    height, width = gray.shape
    image_area = height * width
    mask = np.zeros((height, width), dtype=np.uint8)
    region_count = 0
    try:
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
    except Exception:
        return mask.astype(np.float32), 0
    for raw_region in regions[:8000]:
        region = np.asarray(raw_region).reshape(-1, 2)
        if region.size == 0:
            continue
        xs = region[:, 0]
        ys = region[:, 1]
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        area = int(region.shape[0])
        box_area = max(1, (x1 - x0) * (y1 - y0))
        fill = area / box_area
        if area < 6 or area > image_area * 0.18:
            continue
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        if fill > 0.88 and area > image_area * 0.01:
            continue
        mask[ys, xs] = 255
        region_count += 1
    if region_count:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask.astype(np.float32) / 255.0, region_count


def _filter_components(confidence: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    candidate = (confidence >= 0.33).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    height, width = candidate.shape
    image_area = height * width
    filtered = np.zeros_like(confidence, dtype=np.float32)
    kept = 0
    rejected = 0
    repeated_sizes: dict[tuple[int, int], int] = {}
    large_components = 0
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_w = int(stats[label, cv2.CC_STAT_WIDTH])
        component_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        box_area = max(1, component_w * component_h)
        fill = area / box_area
        component = labels == label
        component_confidence = float(np.mean(confidence[component]))
        if area < 4:
            rejected += 1
            continue
        # Logo/watermark lớn không bị loại theo ngưỡng 3.5%; chỉ chặn vùng đặc cực lớn.
        if area > image_area * 0.32 and fill > 0.80 and component_confidence < 0.74:
            rejected += 1
            large_components += 1
            continue
        if fill > 0.94 and area > image_area * 0.025 and component_confidence < 0.68:
            rejected += 1
            continue
        filtered[component] = np.maximum(filtered[component], confidence[component])
        kept += 1
        bucket = (max(1, round(component_w / 4) * 4), max(1, round(component_h / 4) * 4))
        repeated_sizes[bucket] = repeated_sizes.get(bucket, 0) + 1
    repeating_groups = sum(1 for value in repeated_sizes.values() if value >= 3)
    diagnostics = {
        "components_total": int(max(0, count - 1)),
        "components_kept": kept,
        "components_rejected": rejected,
        "large_components_rejected": large_components,
        "repeating_groups": repeating_groups,
    }
    return filtered, diagnostics


def detect_watermark(rgb: np.ndarray) -> WatermarkDetection:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Ảnh watermark phải là RGB")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    luminance = _multi_scale_residual(gray)
    color = _color_overlay_score(rgb)
    edges = _edge_score(gray)
    text_regions, text_region_count = _mser_text_score(gray)
    raw = np.maximum.reduce(
        [
            luminance * 0.62 + edges * 0.18,
            color * 0.58 + edges * 0.16,
            text_regions * 0.70 + luminance * 0.25,
        ]
    )
    raw = cv2.GaussianBlur(np.clip(raw, 0.0, 1.0), (0, 0), sigmaX=0.65)
    confidence, component_diagnostics = _filter_components(raw.astype(np.float32))
    if np.any(confidence > 0):
        confidence = cv2.GaussianBlur(confidence, (0, 0), sigmaX=0.45)
    selected = confidence >= 0.33
    diagnostics: dict[str, Any] = {
        "algorithm_version": "watermark-detector-v2-classical",
        "pixels": int(np.count_nonzero(selected)),
        "bounds": list(bounds_from_mask(selected, 0)),
        "confidence_mean": round(float(np.mean(confidence[selected])) if np.any(selected) else 0.0, 6),
        "confidence_max": round(float(np.max(confidence)) if confidence.size else 0.0, 6),
        "low_confidence": bool(np.any(selected) and float(np.mean(confidence[selected])) < 0.48),
        "text_region_count": text_region_count,
        "signals": {
            "luminance": round(float(np.mean(luminance)), 6),
            "color": round(float(np.mean(color)), 6),
            "edge": round(float(np.mean(edges)), 6),
        },
        **component_diagnostics,
    }
    return WatermarkDetection(confidence=np.clip(confidence, 0.0, 1.0).astype(np.float32), diagnostics=diagnostics)
