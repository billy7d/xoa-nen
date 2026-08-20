from __future__ import annotations

from typing import Any

import cv2
import numpy as np


EXPAND_PIXELS = {
    "OFF": 0,
    "LOW": 2,
    "MEDIUM": 5,
    "HIGH": 9,
}


def bounds_from_mask(mask: np.ndarray, threshold: float | int = 0) -> tuple[int, int, int, int]:
    values = np.asarray(mask)
    ys, xs = np.nonzero(values > threshold)
    if not xs.size:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def clamp_points(
    points: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    width: int,
    height: int,
) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for item in points:
        if not isinstance(item, dict):
            raise ValueError("points có phần tử không hợp lệ")
        try:
            x = float(item["x"])
            y = float(item["y"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("points có tọa độ không hợp lệ") from error
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("points có tọa độ không hữu hạn")
        result.append(
            {
                "x": round(min(width - 0.5, max(0.5, x)), 3),
                "y": round(min(height - 0.5, max(0.5, y)), 3),
            }
        )
    return result


def simplify_points(points: list[dict[str, float]], minimum_distance: float = 0.65) -> list[dict[str, float]]:
    if len(points) <= 2:
        return points
    simplified = [points[0]]
    last_x = float(points[0]["x"])
    last_y = float(points[0]["y"])
    minimum_sq = minimum_distance * minimum_distance
    for point in points[1:-1]:
        x = float(point["x"])
        y = float(point["y"])
        if (x - last_x) ** 2 + (y - last_y) ** 2 >= minimum_sq:
            simplified.append(point)
            last_x = x
            last_y = y
    if simplified[-1] != points[-1]:
        simplified.append(points[-1])
    return simplified


def _stroke_roi(
    shape: tuple[int, int],
    points: list[dict[str, float]],
    radius: float,
    feather: float,
) -> tuple[int, int, int, int]:
    height, width = shape
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    margin = int(np.ceil(max(1.0, radius) + max(0.0, feather) + 4.0))
    x0 = max(0, int(np.floor(min(xs))) - margin)
    y0 = max(0, int(np.floor(min(ys))) - margin)
    x1 = min(width, int(np.ceil(max(xs))) + margin + 1)
    y1 = min(height, int(np.ceil(max(ys))) + margin + 1)
    return x0, y0, x1, y1


def rasterize_stroke(
    shape: tuple[int, int],
    points: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    radius: float,
    hardness: float = 1.0,
    feather: float = 0.0,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = shape
    clamped = simplify_points(clamp_points(list(points), width, height))
    coverage = np.zeros(shape, dtype=np.float32)
    if not clamped:
        return coverage, (0, 0, 0, 0)
    radius_px = max(1.0, min(float(radius), float(max(width, height))))
    feather_px = max(0.0, min(float(feather), 128.0))
    hardness_value = float(np.clip(hardness, 0.0, 1.0))
    x0, y0, x1, y1 = _stroke_roi(shape, clamped, radius_px, feather_px)
    local_height = y1 - y0
    local_width = x1 - x0
    centerline = np.zeros((local_height, local_width), dtype=np.uint8)
    local_points = [
        (int(round(float(point["x"]) - x0)), int(round(float(point["y"]) - y0)))
        for point in clamped
    ]
    previous: tuple[int, int] | None = None
    for point in local_points:
        cv2.circle(centerline, point, 1, 255, thickness=-1, lineType=cv2.LINE_AA)
        if previous is not None:
            cv2.line(centerline, previous, point, 255, thickness=1, lineType=cv2.LINE_AA)
        previous = point

    distance = cv2.distanceTransform((centerline == 0).astype(np.uint8), cv2.DIST_L2, 3)
    core_radius = radius_px * hardness_value
    if radius_px <= core_radius + 1e-3:
        local = (distance <= radius_px).astype(np.float32)
    else:
        local = np.clip((radius_px - distance) / max(1e-3, radius_px - core_radius), 0.0, 1.0)
        local[distance <= core_radius] = 1.0
    if feather_px > 0:
        # Feather chỉ làm mềm mask, không blur ảnh kết quả.
        local = cv2.GaussianBlur(local, (0, 0), sigmaX=max(0.35, feather_px / 2.5))
        local = np.clip(local, 0.0, 1.0)
    coverage[y0:y1, x0:x1] = np.maximum(coverage[y0:y1, x0:x1], local)
    return coverage, (x0, y0, x1, y1)


def apply_stroke_to_mask(
    current: np.ndarray,
    points: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    radius: float,
    hardness: float,
    feather: float,
    mode: str,
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    if current.ndim != 2:
        raise ValueError("Mask watermark phải là ma trận 2D")
    stroke, stroke_bounds = rasterize_stroke(current.shape, points, radius, hardness, feather)
    before = current.copy()
    normalized_mode = str(mode).upper()
    if normalized_mode in {"ADD", "PLUS", "INCLUDE"}:
        updated = np.maximum(current, stroke)
    elif normalized_mode in {"SUBTRACT", "REMOVE", "MINUS"}:
        updated = current * (1.0 - stroke)
    else:
        raise ValueError("Chế độ cọ watermark không hợp lệ")
    updated = np.clip(updated, 0.0, 1.0).astype(np.float32)
    changed = np.abs(updated - before) > 1e-4
    return updated, bounds_from_mask(changed, 0), int(np.count_nonzero(changed))


def confidence_to_soft_mask(
    confidence: np.ndarray,
    feather: float = 8.0,
    expand: str = "MEDIUM",
) -> np.ndarray:
    values = np.asarray(confidence, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Confidence mask phải là ma trận 2D")
    soft = np.clip((values - 0.25) / 0.65, 0.0, 1.0)
    expand_px = EXPAND_PIXELS.get(str(expand).upper(), EXPAND_PIXELS["MEDIUM"])
    if expand_px > 0 and np.any(soft > 0.05):
        kernel_size = expand_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        soft = cv2.dilate(soft, kernel, iterations=1)
    feather_px = max(0.0, min(float(feather), 80.0))
    if feather_px > 0 and np.any(soft > 0):
        soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=max(0.35, feather_px / 2.5))
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def hard_mask(mask: np.ndarray, threshold: float = 0.35) -> np.ndarray:
    return (np.asarray(mask, dtype=np.float32) >= threshold).astype(np.uint8) * 255
