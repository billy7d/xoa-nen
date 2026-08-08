from __future__ import annotations

from typing import Any

import numpy as np

from .processor import magic_wand_selection


def _interpolated_points(points: list[dict[str, float]], spacing: float) -> list[tuple[float, float]]:
    if not points:
        return []
    result = [(float(points[0]["x"]), float(points[0]["y"]))]
    for previous, current in zip(points, points[1:]):
        x0, y0 = float(previous["x"]), float(previous["y"])
        x1, y1 = float(current["x"]), float(current["y"])
        distance = float(np.hypot(x1 - x0, y1 - y0))
        steps = max(1, int(np.ceil(distance / max(0.5, spacing))))
        for index in range(1, steps + 1):
            ratio = index / steps
            result.append((x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio))
    return result


def apply_brush(
    alpha: np.ndarray,
    source_alpha: np.ndarray,
    locks: np.ndarray,
    points: list[dict[str, float]],
    radius: float,
    hardness: float,
    opacity: float,
    mode: str,
    target_alpha: float = 1.0,
) -> tuple[np.ndarray, tuple[int, int, int, int], dict[str, Any]]:
    if not points:
        raise ValueError("Brush cần ít nhất một point")
    radius = max(0.5, float(radius))
    hardness = float(np.clip(hardness, 0.0, 1.0))
    opacity = float(np.clip(opacity, 0.0, 1.0))
    height, width = alpha.shape
    samples = _interpolated_points(points, max(0.5, radius * 0.25))
    xs = [point[0] for point in samples]
    ys = [point[1] for point in samples]
    x0 = max(0, int(np.floor(min(xs) - radius - 1)))
    y0 = max(0, int(np.floor(min(ys) - radius - 1)))
    x1 = min(width, int(np.ceil(max(xs) + radius + 1)))
    y1 = min(height, int(np.ceil(max(ys) + radius + 1)))

    result = alpha.copy()
    region = result[y0:y1, x0:x1]
    coverage = np.zeros(region.shape, dtype=np.float32)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    for cx, cy in samples:
        # Canonical coordinates address pixel centers at (x + 0.5, y + 0.5).
        normalized = np.sqrt((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2) / radius
        dab = np.where(
            normalized <= hardness,
            1.0,
            np.clip((1.0 - normalized) / max(1e-6, 1.0 - hardness), 0.0, 1.0),
        ).astype(np.float32)
        coverage = np.maximum(coverage, dab)
    coverage *= opacity
    unlocked = locks[y0:y1, x0:x1] == 0
    coverage *= unlocked

    if mode == "keep":
        region[:] = region + coverage * (source_alpha[y0:y1, x0:x1] - region)
    elif mode == "remove":
        region[:] = region * (1.0 - coverage)
    elif mode == "alpha":
        target = np.clip(target_alpha, 0.0, 1.0) * source_alpha[y0:y1, x0:x1]
        region[:] = region + coverage * (target - region)
    else:
        raise ValueError(f"Brush mode không hợp lệ: {mode}")

    result[y0:y1, x0:x1] = np.minimum(
        np.clip(region, 0.0, 1.0), source_alpha[y0:y1, x0:x1]
    )
    return result, (x0, y0, x1, y1), {
        "tool": "brush",
        "mode": mode,
        "radius": radius,
        "hardness": hardness,
        "opacity": opacity,
        "target_alpha": target_alpha,
        "points": points,
        "algorithm_version": "brush-native-v1",
    }


def apply_magic_wand(
    rgb: np.ndarray,
    alpha: np.ndarray,
    source_alpha: np.ndarray,
    locks: np.ndarray,
    x: int,
    y: int,
    tolerance: float,
    softness: float,
    contiguous: bool,
    mode: str,
) -> tuple[np.ndarray, tuple[int, int, int, int], dict[str, Any]]:
    selection = magic_wand_selection(rgb, x, y, tolerance, softness, contiguous)
    selection *= locks == 0
    result = alpha.copy()
    if mode == "remove":
        result *= 1.0 - selection
    elif mode == "keep":
        result += selection * (source_alpha - result)
    else:
        raise ValueError("Magic Wand mode phải là keep hoặc remove")
    result = np.minimum(np.clip(result, 0.0, 1.0), source_alpha)
    ys, xs = np.nonzero(selection > 0.001)
    if xs.size:
        bounds = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    else:
        bounds = (0, 0, 0, 0)
    operation = {
        "tool": "magic_wand",
        "mode": mode,
        "x": int(x),
        "y": int(y),
        "tolerance": float(tolerance),
        "softness": float(softness),
        "contiguous": bool(contiguous),
        "algorithm_version": "magic-wand-v1",
    }
    return result, bounds, operation
