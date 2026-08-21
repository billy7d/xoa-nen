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


def _sparkle_template(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Tạo silhouette bốn cánh kiểu Gemini mà không phụ thuộc asset bên ngoài."""
    template_size = max(16, int(size))
    angles = np.linspace(0.0, np.pi * 2.0, 720, endpoint=False)
    radius = max(4.0, (template_size - 7.0) / 2.0)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    xs = template_size / 2.0 + radius * np.sign(cosine) * np.abs(cosine) ** 3
    ys = template_size / 2.0 + radius * np.sign(sine) * np.abs(sine) ** 3
    points = np.stack((xs, ys), axis=1).round().astype(np.int32)
    fill = np.zeros((template_size, template_size), dtype=np.uint8)
    cv2.fillPoly(fill, [points], 255)
    outline = cv2.morphologyEx(fill, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return fill, outline


def _detect_gemini_sparkle(rgb: np.ndarray) -> WatermarkDetection | None:
    """Nhận diện logo sparkle đáy-phải bằng hình học, chỉ nhận khi match tách biệt rõ."""
    height, width = rgb.shape[:2]
    shortest = min(height, width)
    minimum_size = max(24, int(round(shortest * 0.035)))
    maximum_size = min(160, int(round(shortest * 0.12)))
    if maximum_size < minimum_size:
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    median = float(np.median(gray))
    lower = int(np.clip(median * 0.30, 24, 80))
    upper = int(np.clip(median * 0.90, 90, 190))
    edges = cv2.Canny(gray, lower, upper).astype(np.float32) / 255.0
    search_x0 = int(round(width * 0.55))
    search_y0 = int(round(height * 0.55))
    search = edges[search_y0:, search_x0:]
    candidates: list[tuple[float, int, int, int]] = []
    for size in range(minimum_size, maximum_size + 1, 2):
        if size > search.shape[0] or size > search.shape[1]:
            continue
        _, outline = _sparkle_template(size)
        scores = cv2.matchTemplate(
            search,
            outline.astype(np.float32) / 255.0,
            cv2.TM_CCOEFF_NORMED,
        )
        _, maximum, _, location = cv2.minMaxLoc(scores)
        candidates.append(
            (float(maximum), size, int(location[0] + search_x0), int(location[1] + search_y0))
        )
    if not candidates:
        return None
    candidates.sort(reverse=True)
    best_score, best_size, best_x, best_y = candidates[0]
    best_center = (best_x + best_size / 2.0, best_y + best_size / 2.0)
    distinct_scores = [
        score
        for score, size, x, y in candidates[1:]
        if np.hypot(
            x + size / 2.0 - best_center[0],
            y + size / 2.0 - best_center[1],
        )
        >= max(best_size, size) * 0.65
    ]
    runner_up = max(distinct_scores, default=0.0)
    score_gap = best_score - runner_up
    if best_score < 0.31 or score_gap < 0.035:
        return None

    fill, _ = _sparkle_template(best_size)
    confidence = np.zeros((height, width), dtype=np.float32)
    confidence[best_y : best_y + best_size, best_x : best_x + best_size] = (
        fill.astype(np.float32) / 255.0
    )
    return WatermarkDetection(
        confidence=confidence,
        diagnostics={
            "algorithm_version": "watermark-detector-v3-gemini-sparkle",
            "detector": "GEMINI_SPARKLE_NCC",
            "pixels": int(np.count_nonzero(confidence > 0.01)),
            "bounds": [best_x, best_y, best_x + best_size, best_y + best_size],
            "confidence_mean": round(best_score, 6),
            "confidence_max": round(best_score, 6),
            "low_confidence": False,
            "template_size": best_size,
            "runner_up": round(runner_up, 6),
            "score_gap": round(score_gap, 6),
        },
    )


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
    sparkle = _detect_gemini_sparkle(rgb)
    if sparkle is not None:
        return sparkle
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
    selected_ratio = float(np.mean(selected))
    safety_rejected = selected_ratio > 0.06
    if safety_rejected:
        # Auto-mask sai không được phép lấp một phần lớn ảnh; người dùng vẫn có thể vẽ brush.
        confidence = np.zeros_like(confidence, dtype=np.float32)
        selected = confidence >= 0.33
    diagnostics: dict[str, Any] = {
        "algorithm_version": "watermark-detector-v3-conservative",
        "detector": "CLASSICAL_CONSERVATIVE",
        "pixels": int(np.count_nonzero(selected)),
        "bounds": list(bounds_from_mask(selected, 0)),
        "confidence_mean": round(float(np.mean(confidence[selected])) if np.any(selected) else 0.0, 6),
        "confidence_max": round(float(np.max(confidence)) if confidence.size else 0.0, 6),
        "low_confidence": bool(np.any(selected) and float(np.mean(confidence[selected])) < 0.48),
        "candidate_ratio": round(selected_ratio, 6),
        "safety_rejected": safety_rejected,
        "text_region_count": text_region_count,
        "signals": {
            "luminance": round(float(np.mean(luminance)), 6),
            "color": round(float(np.mean(color)), 6),
            "edge": round(float(np.mean(edges)), 6),
        },
        **component_diagnostics,
    }
    return WatermarkDetection(confidence=np.clip(confidence, 0.0, 1.0).astype(np.float32), diagnostics=diagnostics)
