from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from PIL import Image


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


def flood_connected(eligible: np.ndarray, seeds: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Original V1 scanline flood fill. Keep this implementation frozen."""
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


def _distance_chunks(rgb: np.ndarray, target: np.ndarray, chunk_rows: int = 512) -> np.ndarray:
    height, width = rgb.shape[:2]
    result = np.empty((height, width), dtype=np.float32)
    target = target.astype(np.float32)
    for y0 in range(0, height, chunk_rows):
        chunk = rgb[y0 : y0 + chunk_rows].astype(np.float32)
        delta = chunk - target
        result[y0 : y0 + chunk.shape[0]] = np.sqrt(np.sum(delta * delta, axis=2))
    return result


def artwork_alpha(
    rgb: np.ndarray,
    source_alpha: np.ndarray,
    tolerance: float = 30.0,
    softness: float = 18.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bit-for-bit V1 implementation copied from Git HEAD ed9be77."""
    background = estimate_background_color(rgb)
    proxy_rgb, _, _ = _proxy(rgb, 1024)
    proxy_distance = _distance_chunks(proxy_rgb, background, 256)
    high = max(1.0, tolerance + softness)
    eligible = proxy_distance <= high
    connected_proxy = flood_connected(eligible)
    connected_native = np.asarray(
        Image.fromarray(connected_proxy.astype(np.uint8) * 255, "L").resize(
            (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST
        )
    ) > 0

    distance = _distance_chunks(rgb, background)
    alpha_by_color = smoothstep(max(0.0, tolerance - softness), tolerance + softness, distance)
    alpha = np.where(connected_native, alpha_by_color, 1.0).astype(np.float32)
    alpha *= source_alpha

    diagnostics = {
        "engine": "classical-artwork-v1",
        "background_rgb": [round(float(value), 3) for value in background],
        "tolerance": float(tolerance),
        "softness": float(softness),
        "connected_background_fraction": float(np.mean(connected_native)),
    }
    return alpha, diagnostics


def magic_wand_selection(
    rgb: np.ndarray,
    x: int,
    y: int,
    tolerance: float,
    softness: float,
    contiguous: bool,
) -> np.ndarray:
    """Bit-for-bit V1 Magic Wand implementation."""
    height, width = rgb.shape[:2]
    x = max(0, min(width - 1, int(x)))
    y = max(0, min(height - 1, int(y)))
    target = rgb[y, x].astype(np.float32)
    proxy_rgb, sx, sy = _proxy(rgb, 1024)
    proxy_distance = _distance_chunks(proxy_rgb, target, 256)
    high = max(1.0, tolerance + softness)
    eligible = proxy_distance <= high
    if contiguous:
        px = max(0, min(proxy_rgb.shape[1] - 1, int(x / sx)))
        py = max(0, min(proxy_rgb.shape[0] - 1, int(y / sy)))
        membership = flood_connected(eligible, [(px, py)])
    else:
        membership = eligible
    membership_native = np.asarray(
        Image.fromarray(membership.astype(np.uint8) * 255, "L").resize(
            (width, height), Image.Resampling.NEAREST
        )
    ) > 0
    distance_native = _distance_chunks(rgb, target)
    coverage = 1.0 - smoothstep(
        max(0.0, tolerance - softness), tolerance + softness, distance_native
    )
    return np.where(membership_native, coverage, 0.0).astype(np.float32)
