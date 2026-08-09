from __future__ import annotations

from collections import deque
import time
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .legacy_v1 import artwork_alpha as legacy_artwork_alpha
from .legacy_v1 import magic_wand_selection as legacy_magic_wand_selection

try:  # The release bundle includes opencv-python-headless; source runs keep a safe fallback.
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - exercised by the dependency-fallback test.
    cv2 = None


_LAB_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (values >= edge1).astype(np.float32)
    scaled = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (scaled * scaled * (3.0 - 2.0 * scaled)).astype(np.float32)


def border_pixels(rgb: np.ndarray, thickness: int | None = None) -> np.ndarray:
    height, width = rgb.shape[:2]
    thickness = thickness or max(1, min(height, width) // 100)
    thickness = min(thickness, height // 2 or 1, width // 2 or 1)
    return np.concatenate(
        [
            rgb[:thickness].reshape(-1, 3),
            rgb[-thickness:].reshape(-1, 3),
            rgb[thickness:-thickness, :thickness].reshape(-1, 3)
            if height > 2 * thickness
            else np.empty((0, 3), dtype=rgb.dtype),
            rgb[thickness:-thickness, -thickness:].reshape(-1, 3)
            if height > 2 * thickness
            else np.empty((0, 3), dtype=rgb.dtype),
        ],
        axis=0,
    )


def estimate_background_color(rgb: np.ndarray) -> np.ndarray:
    """Legacy single-colour estimate retained for export decontamination."""
    samples = border_pixels(rgb)
    return np.median(samples.astype(np.float32), axis=0)


def _proxy(rgb: np.ndarray, max_edge: int = 1024) -> tuple[np.ndarray, float, float]:
    height, width = rgb.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    if target == (width, height):
        return rgb, 1.0, 1.0
    resized = np.asarray(Image.fromarray(rgb, "RGB").resize(target, Image.Resampling.BILINEAR))
    return resized, width / target[0], height / target[1]


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an inference sRGB buffer to CIE L*a*b* (D65), without ICC side effects."""
    srgb = rgb.astype(np.float32) / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4),
    )
    xyz = np.empty_like(linear, dtype=np.float32)
    xyz[..., 0] = (
        linear[..., 0] * 0.4124564
        + linear[..., 1] * 0.3575761
        + linear[..., 2] * 0.1804375
    )
    xyz[..., 1] = (
        linear[..., 0] * 0.2126729
        + linear[..., 1] * 0.7151522
        + linear[..., 2] * 0.0721750
    )
    xyz[..., 2] = (
        linear[..., 0] * 0.0193339
        + linear[..., 1] * 0.1191920
        + linear[..., 2] * 0.9503041
    )
    xyz /= _LAB_D65
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3.0 * delta * delta) + 4.0 / 29.0,
    )
    lab = np.empty_like(transformed, dtype=np.float32)
    lab[..., 0] = 116.0 * transformed[..., 1] - 16.0
    lab[..., 1] = 500.0 * (transformed[..., 0] - transformed[..., 1])
    lab[..., 2] = 200.0 * (transformed[..., 1] - transformed[..., 2])
    return lab


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIE L*a*b* (D65) centres back to display sRGB for diagnostics/export."""
    lab = np.asarray(lab, dtype=np.float32)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    delta = 6.0 / 29.0

    def inverse_f(values: np.ndarray) -> np.ndarray:
        return np.where(
            values > delta,
            values**3,
            3.0 * delta * delta * (values - 4.0 / 29.0),
        )

    xyz = np.stack((inverse_f(fx), inverse_f(fy), inverse_f(fz)), axis=-1) * _LAB_D65
    linear = np.empty_like(xyz, dtype=np.float32)
    linear[..., 0] = xyz[..., 0] * 3.2404542 - xyz[..., 1] * 1.5371385 - xyz[..., 2] * 0.4985314
    linear[..., 1] = -xyz[..., 0] * 0.9692660 + xyz[..., 1] * 1.8760108 + xyz[..., 2] * 0.0415560
    linear[..., 2] = xyz[..., 0] * 0.0556434 - xyz[..., 1] * 0.2040259 + xyz[..., 2] * 1.0572252
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb * 255.0, 0.0, 255.0).astype(np.float32)


def _delta_e_76(lab: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = lab - target.astype(np.float32)
    return np.sqrt(np.sum(delta * delta, axis=-1)).astype(np.float32)


def _delta_e_2000(lab: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Vectorised CIEDE2000 using the kL=kC=kH=1 reference conditions."""
    l1, a1, b1 = (lab[..., index].astype(np.float64) for index in range(3))
    l2, a2, b2 = (float(target[index]) for index in range(3))
    c1 = np.hypot(a1, b1)
    c2 = float(np.hypot(a2, b2))
    c_bar = (c1 + c2) * 0.5
    c_bar7 = np.power(c_bar, 7)
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0**7)))
    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    c1_prime = np.hypot(a1_prime, b1)
    c2_prime = np.hypot(a2_prime, b2)
    h1_prime = np.mod(np.degrees(np.arctan2(b1, a1_prime)), 360.0)
    h2_prime = np.mod(np.degrees(np.arctan2(b2, a2_prime)), 360.0)
    h1_prime = np.where(c1_prime == 0.0, 0.0, h1_prime)
    h2_prime = np.where(c2_prime == 0.0, 0.0, h2_prime)

    delta_l_prime = l1 - l2
    delta_c_prime = c1_prime - c2_prime
    delta_h = h1_prime - h2_prime
    delta_h = np.where(c1_prime * c2_prime == 0.0, 0.0, delta_h)
    delta_h = np.where(delta_h > 180.0, delta_h - 360.0, delta_h)
    delta_h = np.where(delta_h < -180.0, delta_h + 360.0, delta_h)
    delta_h_prime = 2.0 * np.sqrt(c1_prime * c2_prime) * np.sin(np.radians(delta_h * 0.5))

    l_bar_prime = (l1 + l2) * 0.5
    c_bar_prime = (c1_prime + c2_prime) * 0.5
    h_sum = h1_prime + h2_prime
    h_diff = np.abs(h1_prime - h2_prime)
    h_bar_prime = np.where(
        c1_prime * c2_prime == 0.0,
        h_sum,
        np.where(
            h_diff <= 180.0,
            h_sum * 0.5,
            np.where(h_sum < 360.0, (h_sum + 360.0) * 0.5, (h_sum - 360.0) * 0.5),
        ),
    )
    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_prime - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_prime))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_prime + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_prime - 63.0))
    )
    delta_theta = 30.0 * np.exp(-np.square((h_bar_prime - 275.0) / 25.0))
    c_bar_prime7 = np.power(c_bar_prime, 7)
    r_c = 2.0 * np.sqrt(c_bar_prime7 / (c_bar_prime7 + 25.0**7))
    s_l = 1.0 + 0.015 * np.square(l_bar_prime - 50.0) / np.sqrt(
        20.0 + np.square(l_bar_prime - 50.0)
    )
    s_c = 1.0 + 0.045 * c_bar_prime
    s_h = 1.0 + 0.015 * c_bar_prime * t
    r_t = -np.sin(np.radians(2.0 * delta_theta)) * r_c
    l_term = delta_l_prime / s_l
    c_term = delta_c_prime / s_c
    h_term = delta_h_prime / s_h
    distance = np.sqrt(
        np.maximum(0.0, l_term * l_term + c_term * c_term + h_term * h_term + r_t * c_term * h_term)
    )
    return distance.astype(np.float32)


def _distance_to_palette_lab(
    lab: np.ndarray,
    palette: np.ndarray,
    metric: str,
) -> np.ndarray:
    result = np.full(lab.shape[:2], np.inf, dtype=np.float32)
    distance_function = _delta_e_2000 if metric == "ciede2000" else _delta_e_76
    for target in palette:
        np.minimum(result, distance_function(lab, target), out=result)
    return result


def _perceptual_distance_chunks(
    rgb: np.ndarray,
    palette: np.ndarray,
    metric: str,
    chunk_rows: int = 256,
) -> np.ndarray:
    height, width = rgb.shape[:2]
    result = np.empty((height, width), dtype=np.float32)
    # CIEDE2000 has several float64 intermediates. Bound each working chunk by
    # pixel count so a 10k-wide guaranteed input cannot create a multi-GB spike.
    bounded_rows = max(8, min(chunk_rows, 300_000 // max(1, width)))
    for y0 in range(0, height, bounded_rows):
        chunk = _srgb_to_lab(rgb[y0 : y0 + bounded_rows])
        result[y0 : y0 + chunk.shape[0]] = _distance_to_palette_lab(chunk, palette, metric)
    return result


def _background_palette(lab: np.ndarray, quality_preset: str) -> tuple[np.ndarray, list[float]]:
    thickness = max(2, min(lab.shape[:2]) // 80)
    sample_groups = [
        lab[:thickness].reshape(-1, 3),
        lab[-thickness:].reshape(-1, 3),
        lab[thickness:-thickness, :thickness].reshape(-1, 3),
        lab[thickness:-thickness, -thickness:].reshape(-1, 3),
    ]
    sampled_groups: list[np.ndarray] = []
    sampled_sides: list[np.ndarray] = []
    for side, group in enumerate(sample_groups):
        if group.shape[0] > 3000:
            indices = np.linspace(0, group.shape[0] - 1, 3000, dtype=np.int64)
            group = group[indices]
        sampled_groups.append(group.astype(np.float32))
        sampled_sides.append(np.full(group.shape[0], side, dtype=np.int8))
    samples = np.concatenate(sampled_groups, axis=0)
    sample_sides = np.concatenate(sampled_sides)
    cluster_limit = 5 if quality_preset == "MAX" else 4 if quality_preset == "QUALITY" else 3
    cluster_count = max(1, min(cluster_limit, samples.shape[0]))
    centers = [np.median(samples, axis=0)]
    closest = _delta_e_76(samples, centers[0])
    for _ in range(1, cluster_count):
        next_index = int(np.argmax(closest))
        if closest[next_index] < 1.0:
            break
        centers.append(samples[next_index].copy())
        closest = np.minimum(closest, _delta_e_76(samples, centers[-1]))
    palette = np.asarray(centers, dtype=np.float32)
    labels = np.zeros(samples.shape[0], dtype=np.int16)
    for _ in range(10):
        distances = np.stack([_delta_e_76(samples, center) for center in palette], axis=1)
        next_labels = np.argmin(distances, axis=1).astype(np.int16)
        next_centers: list[np.ndarray] = []
        for index in range(palette.shape[0]):
            members = samples[next_labels == index]
            next_centers.append(np.median(members, axis=0) if members.size else palette[index])
        next_palette = np.asarray(next_centers, dtype=np.float32)
        if np.array_equal(labels, next_labels) and np.max(np.abs(next_palette - palette)) < 0.05:
            palette = next_palette
            labels = next_labels
            break
        palette = next_palette
        labels = next_labels

    counts = np.bincount(labels, minlength=palette.shape[0]).astype(np.float64)
    fractions = counts / max(1.0, counts.sum())
    side_support = np.zeros(palette.shape[0], dtype=np.int8)
    for index in range(palette.shape[0]):
        for side in range(4):
            side_size = max(1, int(np.count_nonzero(sample_sides == side)))
            support = int(np.count_nonzero((labels == index) & (sample_sides == side)))
            if support >= max(3, round(side_size * 0.02)):
                side_support[index] += 1
    # A genuine background mode normally occurs on at least two sides. A mode
    # seen on just one side is often foreground touching the canvas edge; accept
    # it only when it dominates the whole border.
    keep = (fractions >= 0.04) & ((side_support >= 2) | (fractions >= 0.32))
    if not np.any(keep):
        keep[int(np.argmax(fractions))] = True
    palette = palette[keep]
    fractions = fractions[keep]
    order = np.argsort(fractions)[::-1]
    return palette[order], [float(value) for value in fractions[order]]


def _ui_thresholds(tolerance: float, softness: float) -> tuple[float, float, float]:
    tolerance_de = max(0.35, float(tolerance) * 0.5)
    softness_de = max(0.0, float(softness) * 0.35)
    return (
        max(0.0, tolerance_de - softness_de),
        tolerance_de + softness_de,
        tolerance_de,
    )


def _edge_strength(lab: np.ndarray) -> np.ndarray:
    dx = np.zeros(lab.shape[:2], dtype=np.float32)
    dy = np.zeros(lab.shape[:2], dtype=np.float32)
    dx[:, 1:-1] = 0.5 * np.sqrt(np.sum(np.square(lab[:, 2:] - lab[:, :-2]), axis=2))
    dy[1:-1] = 0.5 * np.sqrt(np.sum(np.square(lab[2:] - lab[:-2]), axis=2))
    edge = np.hypot(dx, dy).astype(np.float32)
    padded = np.pad(edge, 1, mode="edge")
    thickened = np.zeros_like(edge)
    for y_offset in range(3):
        for x_offset in range(3):
            np.maximum(
                thickened,
                padded[y_offset : y_offset + edge.shape[0], x_offset : x_offset + edge.shape[1]],
                out=thickened,
            )
    return thickened


def _adaptive_edge_limit(edge: np.ndarray, tolerance_de: float) -> float:
    thickness = max(1, min(edge.shape) // 100)
    samples = np.concatenate(
        [
            edge[:thickness].reshape(-1),
            edge[-thickness:].reshape(-1),
            edge[thickness:-thickness, :thickness].reshape(-1),
            edge[thickness:-thickness, -thickness:].reshape(-1),
        ]
    )
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    noise_limit = median + 3.5 * max(0.15, mad)
    return float(np.clip(noise_limit + 0.35, 0.8, max(1.2, tolerance_de * 0.16)))


def _morph_mask(mask: np.ndarray, operation: str, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    size = radius * 2 + 1
    image = Image.fromarray(mask.astype(np.uint8) * 255, "L")
    if operation == "dilate":
        filtered = image.filter(ImageFilter.MaxFilter(size))
    elif operation == "erode":
        filtered = image.filter(ImageFilter.MinFilter(size))
    elif operation == "close":
        filtered = image.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
    else:
        raise ValueError(f"Morphology operation không hợp lệ: {operation}")
    return np.asarray(filtered, dtype=np.uint8) > 0


def _resize_float(values: np.ndarray, target: tuple[int, int], resample: int) -> np.ndarray:
    if target == (values.shape[1], values.shape[0]):
        return values.astype(np.float32, copy=True)
    image = Image.fromarray(values.astype(np.float32), "F")
    return np.asarray(image.resize(target, resample), dtype=np.float32).copy()


def _resize_mask(mask: np.ndarray, target: tuple[int, int], resample: int) -> np.ndarray:
    if target == (mask.shape[1], mask.shape[0]):
        return mask.copy()
    image = Image.fromarray(mask.astype(np.uint8) * 255, "L")
    return np.asarray(image.resize(target, resample), dtype=np.uint8) > 0


def _box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.astype(np.float32, copy=True)
    window = radius * 2 + 1
    padded = np.pad(values.astype(np.float32), radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    total = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    return (total / float(window * window)).astype(np.float32)


def _guided_filter(guidance: np.ndarray, proposal: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
    guidance = guidance.astype(np.float32)
    proposal = proposal.astype(np.float32)
    mean_g = _box_mean(guidance, radius)
    mean_p = _box_mean(proposal, radius)
    correlation_g = _box_mean(guidance * guidance, radius)
    correlation_gp = _box_mean(guidance * proposal, radius)
    variance_g = correlation_g - mean_g * mean_g
    covariance_gp = correlation_gp - mean_g * mean_p
    a = covariance_gp / (variance_g + epsilon)
    b = mean_p - a * mean_g
    return np.clip(_box_mean(a, radius) * guidance + _box_mean(b, radius), 0.0, 1.0)


def flood_connected(eligible: np.ndarray, seeds: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Scanline flood fill over a boolean array."""
    height, width = eligible.shape
    visited = np.zeros((height, width), dtype=bool)
    stack: list[tuple[int, int]] = []
    if seeds is None:
        for x in range(width):
            if eligible[0, x]:
                stack.append((x, 0))
            if height > 1 and eligible[height - 1, x]:
                stack.append((x, height - 1))
        for y in range(1, height - 1):
            if eligible[y, 0]:
                stack.append((0, y))
            if width > 1 and eligible[y, width - 1]:
                stack.append((width - 1, y))
    else:
        stack.extend(seeds)

    while stack:
        x, y = stack.pop()
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        if visited[y, x] or not eligible[y, x]:
            continue

        left = x
        while left > 0 and eligible[y, left - 1] and not visited[y, left - 1]:
            left -= 1
        right = x
        while right + 1 < width and eligible[y, right + 1] and not visited[y, right + 1]:
            right += 1
        visited[y, left : right + 1] = True

        for next_y in (y - 1, y + 1):
            if next_y < 0 or next_y >= height:
                continue
            row = eligible[next_y, left : right + 1] & ~visited[next_y, left : right + 1]
            if not np.any(row):
                continue
            starts = np.flatnonzero(row & ~np.r_[False, row[:-1]])
            for start in starts:
                stack.append((left + int(start), next_y))
    return visited


def _linear_rgb(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float32) / 255.0
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def _field_basis(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    if order == 0:
        return np.ones((x.size, 1), dtype=np.float32)
    if order == 1:
        return np.stack((np.ones_like(x), x, y), axis=1).astype(np.float32)
    return np.stack((np.ones_like(x), x, y, x * y, x * x, y * y), axis=1).astype(
        np.float32
    )


def _fit_background_field(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a robust spatial background model while keeping border provenance."""
    height, width = rgb.shape[:2]
    thickness = max(2, min(height, width) // 60)
    yy, xx = np.mgrid[0:height, 0:width]
    border = (yy < thickness) | (yy >= height - thickness) | (xx < thickness) | (
        xx >= width - thickness
    )
    bx = (xx[border].astype(np.float32) / max(1, width - 1)) * 2.0 - 1.0
    by = (yy[border].astype(np.float32) / max(1, height - 1)) * 2.0 - 1.0
    samples_rgb = rgb[border].astype(np.float32)
    samples_linear = _linear_rgb(rgb)[border]
    if samples_linear.shape[0] > 16000:
        indices = np.linspace(0, samples_linear.shape[0] - 1, 16000, dtype=np.int64)
        bx, by = bx[indices], by[indices]
        samples_rgb, samples_linear = samples_rgb[indices], samples_linear[indices]

    holdout = np.arange(samples_linear.shape[0]) % 5 == 0
    train = ~holdout
    candidates: list[tuple[int, np.ndarray, float]] = []
    for order in (0, 1, 2):
        design = _field_basis(bx, by, order)
        weights = np.ones(int(np.count_nonzero(train)), dtype=np.float32)
        coefficients = np.zeros((design.shape[1], 3), dtype=np.float32)
        for _ in range(5):
            weighted_design = design[train] * weights[:, None]
            weighted_values = samples_linear[train] * weights[:, None]
            coefficients = np.linalg.lstsq(
                weighted_design, weighted_values, rcond=None
            )[0].astype(np.float32)
            residual = np.linalg.norm(design[train] @ coefficients - samples_linear[train], axis=1)
            median = float(np.median(residual))
            mad = float(np.median(np.abs(residual - median)))
            cutoff = median + 2.5 * max(1e-4, 1.4826 * mad)
            weights = np.minimum(1.0, cutoff / np.maximum(residual, 1e-6)).astype(np.float32)
        held = np.linalg.norm(design[holdout] @ coefficients - samples_linear[holdout], axis=1)
        score = float(np.percentile(held, 90)) + order * 0.0008
        candidates.append((order, coefficients, score))

    best_score = min(item[2] for item in candidates)
    order, coefficients, score = next(
        item for item in candidates if item[2] <= best_score * 1.08 + 0.0015
    )
    gx = (xx.astype(np.float32) / max(1, width - 1)) * 2.0 - 1.0
    gy = (yy.astype(np.float32) / max(1, height - 1)) * 2.0 - 1.0
    field_linear = (_field_basis(gx.reshape(-1), gy.reshape(-1), order) @ coefficients).reshape(
        height, width, 3
    )
    field_linear = np.clip(field_linear, 0.0, 1.0)
    field_srgb = np.where(
        field_linear <= 0.0031308,
        field_linear * 12.92,
        1.055 * np.power(field_linear, 1.0 / 2.4) - 0.055,
    )
    median = np.median(samples_rgb, axis=0)
    border_rgb_p95 = float(np.percentile(np.linalg.norm(samples_rgb - median, axis=1), 95))
    return np.rint(np.clip(field_srgb * 255.0, 0.0, 255.0)).astype(np.uint8), {
        "order": int(order),
        "coefficients_linear": [
            [round(float(value), 9) for value in row] for row in coefficients
        ],
        "heldout_linear_p90": round(score, 6),
        "border_rgb_p95": round(border_rgb_p95, 4),
        "border_sample_count": int(samples_rgb.shape[0]),
    }


def _component_boxes(mask: np.ndarray, limit: int = 30) -> list[dict[str, Any]]:
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    components: list[dict[str, Any]] = []
    for start_y, start_x in zip(*np.nonzero(mask & ~visited)):
        if visited[start_y, start_x]:
            continue
        queue = deque([(int(start_x), int(start_y))])
        visited[start_y, start_x] = True
        count = 0
        min_x = max_x = int(start_x)
        min_y = max_y = int(start_y)
        while queue:
            x, y = queue.popleft()
            count += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))
        if count >= 4:
            components.append(
                {
                    "bbox_proxy": [min_x, min_y, max_x + 1, max_y + 1],
                    "area_proxy": count,
                    "reason": "candidate_disagreement",
                }
            )
    components.sort(key=lambda item: item["area_proxy"], reverse=True)
    return components[:limit]


def _graphcut_foreground(
    proxy_rgb: np.ndarray,
    legacy_alpha: np.ndarray,
    spatial_distance: np.ndarray,
    spatial_connected: np.ndarray,
    low: float,
    high: float,
    quality_preset: str,
    semantic_alpha: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    if cv2 is None:
        return legacy_alpha >= 0.5, "opencv_unavailable"
    height, width = legacy_alpha.shape
    mask = np.full((height, width), cv2.GC_PR_FGD, dtype=np.uint8)
    border = max(2, min(height, width) // 80)
    mask[:border] = cv2.GC_BGD
    mask[-border:] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD
    mask[(legacy_alpha <= 0.03) & spatial_connected] = cv2.GC_BGD
    mask[(spatial_distance <= low) & spatial_connected & (legacy_alpha < 0.85)] = cv2.GC_PR_BGD
    far_foreground = (spatial_distance >= high + 5.0) & (legacy_alpha >= 0.98)
    mask[far_foreground] = cv2.GC_FGD

    coverage = float(np.mean(spatial_distance <= high))
    if coverage > 0.90 and not np.any(far_foreground):
        # A palette covering almost the whole image is exactly the destructive V2
        # failure. Seed only a small, interior V1 foreground core; never declare it
        # background without an independent topology vote.
        y0, y1 = height // 3, max(height // 3 + 1, height * 2 // 3)
        x0, x1 = width // 3, max(width // 3 + 1, width * 2 // 3)
        central = np.zeros(mask.shape, dtype=bool)
        central[y0:y1, x0:x1] = True
        residual_floor = float(np.percentile(spatial_distance, 70))
        residual_spread = float(np.percentile(spatial_distance, 90) - np.percentile(spatial_distance, 50))
        core = central & (
            (legacy_alpha >= 0.98)
            | ((spatial_distance >= max(0.8, residual_floor)) & (residual_spread >= 0.5))
        )
        if np.any(core):
            mask[core] = cv2.GC_FGD

    if semantic_alpha is not None:
        semantic = _resize_float(semantic_alpha, (width, height), Image.Resampling.BILINEAR)
        mask[semantic >= 0.97] = cv2.GC_FGD
        mask[(semantic <= 0.03) & spatial_connected] = cv2.GC_BGD
        mask[(semantic >= 0.75) & (mask != cv2.GC_BGD)] = cv2.GC_PR_FGD
        mask[(semantic <= 0.25) & spatial_connected & (mask != cv2.GC_FGD)] = cv2.GC_PR_BGD

    if not np.any((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)):
        return legacy_alpha >= 0.5, "missing_foreground_seed"
    try:
        bg_model = np.zeros((1, 65), dtype=np.float64)
        fg_model = np.zeros((1, 65), dtype=np.float64)
        iterations = 1 if quality_preset == "FAST" else 2
        cv2.setRNGSeed(0)
        cv2.grabCut(
            cv2.cvtColor(proxy_rgb, cv2.COLOR_RGB2BGR),
            mask,
            None,
            bg_model,
            fg_model,
            iterations,
            cv2.GC_INIT_WITH_MASK,
        )
        foreground = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        fraction = float(np.mean(foreground))
        if fraction < 0.002 or fraction > 0.998:
            return legacy_alpha >= 0.5, "degenerate_graphcut"
        return foreground, "ok"
    except Exception as error:  # OpenCV errors must not take down the worker.
        return legacy_alpha >= 0.5, f"graphcut_error:{type(error).__name__}"


def _native_refine_unknown(
    rgb: np.ndarray,
    proposal: np.ndarray,
    unknown: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Tile-bounded native RGB-guided refinement, applied only to unknown pixels."""
    if not np.any(unknown):
        return proposal.astype(np.float32, copy=True)
    linear = _linear_rgb(rgb)
    luminance = (
        linear[..., 0] * 0.2126 + linear[..., 1] * 0.7152 + linear[..., 2] * 0.0722
    ).astype(np.float32)
    result = proposal.astype(np.float32, copy=True)
    height, width = proposal.shape
    tile_size = 512
    margin = radius * 3 + 2
    for y0 in range(0, height, tile_size):
        y1 = min(height, y0 + tile_size)
        for x0 in range(0, width, tile_size):
            x1 = min(width, x0 + tile_size)
            tile_unknown = unknown[y0:y1, x0:x1]
            if not np.any(tile_unknown):
                continue
            cy0, cy1 = max(0, y0 - margin), min(height, y1 + margin)
            cx0, cx1 = max(0, x0 - margin), min(width, x1 + margin)
            refined = _guided_filter(
                luminance[cy0:cy1, cx0:cx1],
                proposal[cy0:cy1, cx0:cx1],
                radius,
                epsilon=0.0008,
            )
            center = refined[y0 - cy0 : y1 - cy0, x0 - cx0 : x1 - cx0]
            view = result[y0:y1, x0:x1]
            view[tile_unknown] = center[tile_unknown]
    return np.clip(result, 0.0, 1.0)


def artwork_alpha(
    rgb: np.ndarray,
    source_alpha: np.ndarray,
    tolerance: float = 30.0,
    softness: float = 18.0,
    quality_preset: str = "QUALITY",
    engine_profile: str = "V3_BALANCED",
    semantic_alpha: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    stage_started = started
    stage_timings: dict[str, float] = {}
    quality_preset = str(quality_preset).upper()
    engine_profile = str(engine_profile).upper()
    if quality_preset not in {"FAST", "QUALITY"}:
        raise ValueError(f"Quality preset không hợp lệ: {quality_preset}")
    if engine_profile not in {"LEGACY_V1", "V3_BALANCED", "V3_AI_LOCAL"}:
        raise ValueError(f"Engine profile không hợp lệ: {engine_profile}")

    legacy_alpha, legacy_diagnostics = legacy_artwork_alpha(
        rgb, source_alpha, tolerance=tolerance, softness=softness
    )
    stage_timings["legacy_native_ms"] = round((time.perf_counter() - stage_started) * 1000.0, 3)
    stage_started = time.perf_counter()
    if engine_profile == "LEGACY_V1":
        diagnostics = dict(legacy_diagnostics)
        diagnostics.update(
            {
                "engine_profile": "LEGACY_V1",
                "selected_strategy": "legacy_v1_exact",
                "quality_preset": quality_preset,
                "needs_review": False,
                "review_regions": [],
                "stage_timings_ms": stage_timings,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        )
        return legacy_alpha, diagnostics

    proxy_edge = 104 if quality_preset == "FAST" else 192
    proxy_rgb, _, _ = _proxy(rgb, proxy_edge)
    proxy_source_alpha = _resize_float(
        source_alpha, (proxy_rgb.shape[1], proxy_rgb.shape[0]), Image.Resampling.BILINEAR
    )
    legacy_proxy, _ = legacy_artwork_alpha(
        proxy_rgb, proxy_source_alpha, tolerance=tolerance, softness=softness
    )
    field_rgb, field_diagnostics = _fit_background_field(proxy_rgb)
    proxy_lab = _srgb_to_lab(proxy_rgb)
    field_lab = _srgb_to_lab(field_rgb)
    spatial_distance = np.sqrt(np.sum(np.square(proxy_lab - field_lab), axis=2)).astype(np.float32)
    low, high, tolerance_de = _ui_thresholds(tolerance, softness)
    spatial_connected = flood_connected(spatial_distance <= high)
    stage_timings["spatial_evidence_ms"] = round((time.perf_counter() - stage_started) * 1000.0, 3)
    stage_started = time.perf_counter()

    flat_limit = max(9.0, float(tolerance) * 0.40)
    is_flat_background = field_diagnostics["order"] == 0 and (
        float(field_diagnostics["border_rgb_p95"]) <= flat_limit
    )
    if is_flat_background and semantic_alpha is None:
        diagnostics = {
            "engine": "hybrid-cutout-v3",
            "engine_profile": engine_profile,
            "selected_strategy": "legacy_v1_exact_flat_gate",
            "fallback_reason": None,
            "quality_preset": quality_preset,
            "tolerance": float(tolerance),
            "softness": float(softness),
            "background_rgb": legacy_diagnostics["background_rgb"],
            "background_palette_rgb": [legacy_diagnostics["background_rgb"]],
            "background_model": field_diagnostics,
            "candidate_scores": {
                "legacy_background_fraction": round(float(np.mean(legacy_proxy < 0.5)), 6),
                "spatial_background_fraction": round(float(np.mean(spatial_connected)), 6),
                "candidate_disagreement_fraction": 0.0,
            },
            "needs_review": False,
            "review_regions": [],
            "stage_timings_ms": stage_timings,
            "ai_models_used": [],
            "source_alpha_contract": "multiply",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        return legacy_alpha, diagnostics

    graph_foreground, graph_status = _graphcut_foreground(
        proxy_rgb,
        legacy_proxy,
        spatial_distance,
        spatial_connected,
        low,
        high,
        quality_preset,
        semantic_alpha,
    )
    stage_timings["graphcut_ms"] = round((time.perf_counter() - stage_started) * 1000.0, 3)
    stage_started = time.perf_counter()
    legacy_background = legacy_proxy < 0.5
    spatial_background = spatial_connected
    graph_background = ~graph_foreground
    background_votes = (
        legacy_background.astype(np.uint8)
        + spatial_background.astype(np.uint8)
        + graph_background.astype(np.uint8)
    )
    palette_coverage = float(np.mean(spatial_distance <= high))
    color_collapse = palette_coverage > 0.90 and (
        float(np.mean(legacy_background)) > 0.90
        or float(np.mean(spatial_background)) > 0.90
    )
    # Recover only V1 pixels inside a one-proxy-pixel support neighbourhood of
    # graph-cut. This is candidate intersection, not foreground expansion: no
    # pixel absent from V1 membership can be made foreground by this guard.
    supported_v1 = (
        (legacy_proxy >= 0.5) & _morph_mask(graph_foreground, "dilate", 2)
        if graph_status == "ok"
        else np.zeros_like(graph_foreground)
    )
    if color_collapse and graph_status == "ok":
        # V1 and spatial Lab are correlated color evidence. When both claim almost
        # the whole image is background, do not let two correlated votes erase a
        # graph-cut foreground (the white tumbler regression).
        majority_foreground = graph_foreground | supported_v1
    elif graph_status == "ok":
        majority_foreground = (background_votes < 2) | supported_v1
    else:
        majority_foreground = legacy_proxy >= 0.5
    unanimous_background = background_votes == 3
    unanimous_foreground = background_votes == 0
    disagreement = ~(unanimous_background | unanimous_foreground)
    boundary = _morph_mask(majority_foreground, "dilate", 1) ^ _morph_mask(
        majority_foreground, "erode", 1
    )
    unknown_proxy = disagreement | boundary
    proposal_proxy = majority_foreground.astype(np.float32)

    target = (rgb.shape[1], rgb.shape[0])
    proposal_native = _resize_float(proposal_proxy, target, Image.Resampling.BILINEAR)
    unknown_native = _resize_mask(unknown_proxy, target, Image.Resampling.NEAREST)
    if quality_preset == "QUALITY":
        proposal_native = _native_refine_unknown(rgb, proposal_native, unknown_native, radius=3)
    sure_background_native = _resize_mask(unanimous_background, target, Image.Resampling.NEAREST)
    sure_foreground_native = _resize_mask(unanimous_foreground, target, Image.Resampling.NEAREST)
    proposal_native[sure_background_native] = 0.0
    proposal_native[sure_foreground_native] = 1.0
    # Multiplication, not min(), is the immutable source-alpha contract.
    alpha = (np.clip(proposal_native, 0.0, 1.0) * source_alpha).astype(np.float32)
    stage_timings["native_refine_ms"] = round((time.perf_counter() - stage_started) * 1000.0, 3)

    review_regions: list[dict[str, Any]] = []
    for region in _component_boxes(disagreement):
        x0, y0, x1, y1 = region.pop("bbox_proxy")
        region["bbox"] = [
            round(x0 * rgb.shape[1] / proxy_rgb.shape[1]),
            round(y0 * rgb.shape[0] / proxy_rgb.shape[0]),
            round(x1 * rgb.shape[1] / proxy_rgb.shape[1]),
            round(y1 * rgb.shape[0] / proxy_rgb.shape[0]),
        ]
        review_regions.append(region)
    disagreement_fraction = float(np.mean(disagreement))
    fallback_reason = None if graph_status == "ok" else graph_status
    diagnostics = {
        "engine": "hybrid-cutout-v3",
        "engine_profile": engine_profile,
        "selected_strategy": "hybrid_consensus_graphcut",
        "fallback_reason": fallback_reason,
        "quality_preset": quality_preset,
        "tolerance": float(tolerance),
        "softness": float(softness),
        "tolerance_delta_e": round(tolerance_de, 4),
        "background_rgb": legacy_diagnostics["background_rgb"],
        "background_palette_rgb": [legacy_diagnostics["background_rgb"]],
        "background_model": field_diagnostics,
        "graphcut_status": graph_status,
        "candidate_scores": {
            "legacy_background_fraction": round(float(np.mean(legacy_background)), 6),
            "spatial_background_fraction": round(float(np.mean(spatial_background)), 6),
            "graph_background_fraction": round(float(np.mean(graph_background)), 6),
            "palette_high_coverage": round(palette_coverage, 6),
            "correlated_color_collapse": color_collapse,
            "candidate_disagreement_fraction": round(disagreement_fraction, 6),
        },
        "needs_review": bool(disagreement_fraction > 0.01 or graph_status != "ok"),
        "review_regions": review_regions,
        "ai_models_used": [],
        "native_unknown_refinement": quality_preset == "QUALITY",
        "source_alpha_contract": "multiply",
        "stage_timings_ms": stage_timings,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    return alpha, diagnostics


def _seed_colour(rgb: np.ndarray, x: int, y: int) -> tuple[np.ndarray, float]:
    radius = max(1, min(4, round(min(rgb.shape[:2]) / 512)))
    x0, x1 = max(0, x - radius), min(rgb.shape[1], x + radius + 1)
    y0, y1 = max(0, y - radius), min(rgb.shape[0], y + radius + 1)
    patch_lab = _srgb_to_lab(rgb[y0:y1, x0:x1]).reshape(-1, 3)
    clicked = _srgb_to_lab(rgb[y : y + 1, x : x + 1])[0, 0]
    clicked_distance = _delta_e_2000(patch_lab, clicked)
    robust_cutoff = min(4.0, max(1.25, float(np.percentile(clicked_distance, 40)) * 1.5))
    same_side = patch_lab[clicked_distance <= robust_cutoff]
    target = np.median(same_side, axis=0).astype(np.float32)
    noise = float(np.median(_delta_e_2000(same_side, target)))
    return target, noise


def _link_connected(
    eligible: np.ndarray,
    distance: np.ndarray,
    lab: np.ndarray,
    seed: tuple[int, int],
    low: float,
    edge_budget: float,
    texture_relax: bool,
) -> np.ndarray:
    """Flood through pixel links; texture is not converted into forbidden pixels."""
    height, width = eligible.shape
    visited = np.zeros((height, width), dtype=bool)
    sx, sy = seed
    if not (0 <= sx < width and 0 <= sy < height) or not eligible[sy, sx]:
        return visited
    queue = deque([(sx, sy)])
    stopped: list[tuple[int, int]] = []
    visited[sy, sx] = True
    while queue:
        x, y = queue.popleft()
        current = lab[y, x]
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if visited[ny, nx] or not eligible[ny, nx]:
                continue
            link_delta = float(np.linalg.norm(lab[ny, nx] - current))
            confident_texture = distance[y, x] <= low and distance[ny, nx] <= low
            allowed = link_delta <= edge_budget or (
                texture_relax and confident_texture and link_delta <= edge_budget * 1.75
            )
            if allowed:
                visited[ny, nx] = True
                queue.append((nx, ny))
            else:
                # Keep one eligible fractional fringe for anti-aliased pixels,
                # but never traverse through it into the region on the far side.
                visited[ny, nx] = True
                stopped.append((nx, ny))
    # Include one more eligible soft-fringe sample so a two-pixel antialias ramp
    # can retain fractional coverage. These pixels are never enqueued, so this
    # cannot flood the object behind a coherent edge.
    for x, y in stopped:
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height and eligible[ny, nx]:
                visited[ny, nx] = True
    return visited


def magic_wand_selection(
    rgb: np.ndarray,
    x: int,
    y: int,
    tolerance: float,
    softness: float,
    contiguous: bool,
    algorithm: str = "SMART",
) -> np.ndarray:
    algorithm = str(algorithm).upper()
    if algorithm == "LEGACY_COLOR":
        return legacy_magic_wand_selection(rgb, x, y, tolerance, softness, contiguous)
    if algorithm != "SMART":
        raise ValueError(f"Wand algorithm không hợp lệ: {algorithm}")
    height, width = rgb.shape[:2]
    x = max(0, min(width - 1, int(x)))
    y = max(0, min(height - 1, int(y)))
    target, seed_noise = _seed_colour(rgb, x, y)
    proxy_rgb, sx, sy = _proxy(rgb, 1024)
    proxy_lab = _srgb_to_lab(proxy_rgb)
    proxy_distance = _delta_e_2000(proxy_lab, target)
    low, high, tolerance_de = _ui_thresholds(tolerance, softness)
    # A small robust patch absorbs JPEG/compression noise without allowing the
    # selection threshold to drift as the flood grows.
    noise_allowance = min(3.0, seed_noise * 1.5)
    proxy_distance = np.maximum(0.0, proxy_distance - noise_allowance)
    eligible = proxy_distance <= high

    px = max(0, min(proxy_rgb.shape[1] - 1, int(x / sx)))
    py = max(0, min(proxy_rgb.shape[0] - 1, int(y / sy)))
    if contiguous:
        edge_budget = max(2.2, tolerance_de * 0.20, seed_noise * 6.0 + 2.0)
        membership_proxy = _link_connected(
            eligible,
            proxy_distance,
            proxy_lab,
            (px, py),
            low,
            edge_budget,
            seed_noise > 0.75,
        )
    else:
        membership_proxy = eligible

    native_distance = _perceptual_distance_chunks(rgb, target[None, :], "ciede2000")
    native_distance = np.maximum(0.0, native_distance - noise_allowance)
    native_eligible = native_distance <= high
    if contiguous:
        candidate_native = _resize_mask(
            membership_proxy, (width, height), Image.Resampling.NEAREST
        )
        membership_native = flood_connected(native_eligible & candidate_native, [(x, y)])
    else:
        membership_native = native_eligible
    coverage = 1.0 - smoothstep(low, high, native_distance)
    # Selection coverage is strictly zero outside membership; no dilation leak.
    return np.where(membership_native, coverage, 0.0).astype(np.float32)


def analyze_components(alpha: np.ndarray, max_edge: int = 512) -> list[dict[str, Any]]:
    height, width = alpha.shape
    scale = min(1.0, max_edge / max(height, width))
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    mask = np.asarray(
        Image.fromarray((alpha > 0.05).astype(np.uint8) * 255, "L").resize(
            target, Image.Resampling.NEAREST
        )
    ) > 0
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[dict[str, Any]] = []
    proxy_h, proxy_w = mask.shape
    area_scale = (width / proxy_w) * (height / proxy_h)

    for start_y, start_x in zip(*np.nonzero(mask & ~visited)):
        if visited[start_y, start_x]:
            continue
        queue = deque([(int(start_x), int(start_y))])
        visited[start_y, start_x] = True
        count = 0
        min_x = max_x = int(start_x)
        min_y = max_y = int(start_y)
        while queue:
            cx, cy = queue.popleft()
            count += 1
            min_x = min(min_x, cx)
            max_x = max(max_x, cx)
            min_y = min(min_y, cy)
            max_y = max(max_y, cy)
            for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                if 0 <= nx < proxy_w and 0 <= ny < proxy_h and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))
        native_area = max(1, round(count * area_scale))
        components.append(
            {
                "id": len(components) + 1,
                "area_px": native_area,
                "bbox": [
                    round(min_x * width / proxy_w),
                    round(min_y * height / proxy_h),
                    round((max_x + 1) * width / proxy_w),
                    round((max_y + 1) * height / proxy_h),
                ],
                "needs_review": native_area < 64,
            }
        )
    components.sort(key=lambda item: item["area_px"], reverse=True)
    return components[:500]


def select_components(
    alpha: np.ndarray,
    selected_ids: set[int],
    max_edge: int = 512,
) -> np.ndarray:
    """Keep selected analyze_components IDs without deleting source component data."""
    height, width = alpha.shape
    scale = min(1.0, max_edge / max(height, width))
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    mask = np.asarray(
        Image.fromarray((alpha > 0.05).astype(np.uint8) * 255, "L").resize(
            target, Image.Resampling.NEAREST
        )
    ) > 0
    visited = np.zeros(mask.shape, dtype=bool)
    selected_proxy = np.zeros(mask.shape, dtype=bool)
    component_id = 0
    proxy_h, proxy_w = mask.shape
    for start_y, start_x in zip(*np.nonzero(mask & ~visited)):
        if visited[start_y, start_x]:
            continue
        component_id += 1
        queue = deque([(int(start_x), int(start_y))])
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            x, y = queue.popleft()
            pixels.append((x, y))
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < proxy_w and 0 <= ny < proxy_h and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))
        if component_id in selected_ids:
            xs = np.fromiter((pixel[0] for pixel in pixels), dtype=np.int32)
            ys = np.fromiter((pixel[1] for pixel in pixels), dtype=np.int32)
            selected_proxy[ys, xs] = True
    selected_native = _resize_mask(selected_proxy, (width, height), Image.Resampling.NEAREST)
    return np.where(selected_native, alpha, 0.0).astype(np.float32)
