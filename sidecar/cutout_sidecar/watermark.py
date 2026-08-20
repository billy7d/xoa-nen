"""Conservative watermark masks and local-first hybrid reconstruction."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from .model_runtime import LocalModelRuntime


def _bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _mask_u8(mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((np.asarray(mask) > 0).astype(np.uint8) * 255)


def _gradient_complexity(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    ring = cv2.dilate(_mask_u8(mask), np.ones((9, 9), np.uint8)) > 0
    ring &= ~(_mask_u8(mask) > 0)
    samples = magnitude[ring] if np.any(ring) else magnitude.reshape(-1)
    edges = cv2.Canny(gray, 55, 130)
    boundary = cv2.morphologyEx(_mask_u8(mask), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return {
        "gradient_mean": round(float(np.mean(samples)), 4),
        "gradient_std": round(float(np.std(samples)), 4),
        "edge_crossings": float(np.count_nonzero((edges > 0) & (boundary > 0))),
        "mask_fraction": round(float(np.mean(mask > 0)), 7),
    }


def automatic_watermark_mask(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Propose a conservative text/logo mask using multi-scale Lab residuals."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Ảnh watermark phải là RGB")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    height, width = lab.shape[:2]
    lightness = lab[:, :, 0]
    chroma = cv2.magnitude(lab[:, :, 1].astype(np.float32) - 128.0, lab[:, :, 2].astype(np.float32) - 128.0)
    short = cv2.GaussianBlur(lightness, (0, 0), sigmaX=max(1.2, min(width, height) / 420))
    broad = cv2.GaussianBlur(lightness, (0, 0), sigmaX=max(3.0, min(width, height) / 90))
    local_residual = cv2.absdiff(lightness, short)
    broad_residual = cv2.absdiff(lightness, broad)
    chroma_residual = cv2.absdiff(chroma, cv2.GaussianBlur(chroma, (0, 0), sigmaX=max(2.0, min(width, height) / 120)))
    median = float(np.median(broad_residual))
    threshold = max(13.0, min(38.0, median + 2.6 * float(np.median(np.abs(broad_residual - median)))))
    residual = ((broad_residual >= threshold) | (local_residual >= threshold * 0.9) | (chroma_residual >= threshold * 0.8)).astype(np.uint8) * 255
    edges = cv2.Canny(lightness, 45, 130)
    candidate = cv2.bitwise_and(residual, cv2.dilate(edges, np.ones((3, 3), np.uint8)))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    selected = np.zeros_like(candidate)
    image_area = width * height
    accepted: list[float] = []
    for label in range(1, count):
        _, _, component_w, component_h, area = stats[label]
        if area < 6 or area > image_area * 0.028 or component_w < 2 or component_h < 2:
            continue
        fill = area / max(1, component_w * component_h)
        aspect = max(component_w, component_h) / max(1, min(component_w, component_h))
        if fill > 0.76 or (fill > 0.58 and aspect < 1.35):
            continue
        selected[labels == label] = 255
        accepted.append(float(fill))
    if np.any(selected):
        distance = cv2.distanceTransform((selected == 0).astype(np.uint8), cv2.DIST_L2, 3)
        selected[distance <= (1.5 if len(accepted) > 14 else 2.2)] = 255
    pixels = int(np.count_nonzero(selected))
    compactness = float(np.mean(accepted)) if accepted else 1.0
    confidence = min(0.99, max(0.0, 0.35 + min(0.35, len(accepted) / 40.0) + (0.22 if pixels else 0.0) - compactness * 0.12))
    return selected, {
        "pixels": pixels, "bounds": list(_bounds(selected > 0)), "component_count": len(accepted),
        "confidence": round(confidence, 3), "needs_review": bool(not pixels or confidence < 0.72),
        "algorithm_version": "watermark-mask-lab-multiscale-v2",
    }


def brush_mask(shape: tuple[int, int], points: list[dict[str, float]], radius: float) -> np.ndarray:
    return apply_mask_stroke(np.zeros(shape, dtype=np.uint8), points, radius, "ADD")


def apply_mask_stroke(mask: np.ndarray, points: list[dict[str, float]], radius: float, mode: str = "ADD") -> np.ndarray:
    """Apply an add/erase brush to an existing staged mask."""
    if mask.ndim != 2:
        raise ValueError("Mask watermark phải là ma trận 2D")
    if not points:
        return _mask_u8(mask)
    operation = str(mode).upper()
    if operation not in {"ADD", "ERASE"}:
        raise ValueError("Mask brush mode phải là ADD hoặc ERASE")
    result, paint = _mask_u8(mask), np.zeros_like(mask, dtype=np.uint8)
    radius_px = max(1, min(int(round(float(radius))), max(mask.shape)))
    previous: tuple[int, int] | None = None
    for point in points:
        center = (int(round(float(point["x"]))), int(round(float(point["y"]))))
        cv2.circle(paint, center, radius_px, 255, thickness=-1, lineType=cv2.LINE_AA)
        if previous is not None:
            cv2.line(paint, previous, center, 255, thickness=radius_px * 2, lineType=cv2.LINE_AA)
        previous = center
    result[paint > 0] = 255 if operation == "ADD" else 0
    return result


def _roi_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _bounds(mask > 0)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0)
    height, width = mask.shape
    padding = max(32, min(256, max(x1 - x0, y1 - y0) * 2))
    return (max(0, x0 - padding), max(0, y0 - padding), min(width, x1 + padding), min(height, y1 + padding))


def _telea(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.inpaint(rgb, _mask_u8(mask), 4, cv2.INPAINT_TELEA)


def _structure_texture(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    x0, y0, x1, y1 = _roi_bounds(mask)
    source, local_mask = rgb[y0:y1, x0:x1], _mask_u8(mask[y0:y1, x0:x1])
    candidate: np.ndarray | None = None
    method = "telea_fallback"
    try:
        xphoto = getattr(cv2, "xphoto", None)
        if xphoto is not None and hasattr(xphoto, "inpaint"):
            lab, output = cv2.cvtColor(source, cv2.COLOR_RGB2LAB), np.empty_like(source)
            xphoto.inpaint(lab, cv2.bitwise_not(local_mask), output, xphoto.INPAINT_SHIFTMAP)
            candidate, method = cv2.cvtColor(output, cv2.COLOR_LAB2RGB), "opencv-xphoto-shiftmap"
    except cv2.error:
        candidate = None
    if candidate is None:
        candidate = _telea(source, local_mask)
    result, selected = rgb.copy(), local_mask > 0
    view = result[y0:y1, x0:x1]
    view[selected] = candidate[selected]
    result[y0:y1, x0:x1] = view
    return result, {"status": "ok", "method": method, "roi": [x0, y0, x1, y1]}


def _choose_engine(rgb: np.ndarray, mask: np.ndarray) -> tuple[str, dict[str, float]]:
    complexity = _gradient_complexity(rgb, mask)
    if complexity["mask_fraction"] <= 0.0012 and complexity["gradient_mean"] < 18.0 and complexity["edge_crossings"] < 8:
        return "FAST", complexity
    if complexity["mask_fraction"] <= 0.045 and (complexity["gradient_mean"] >= 18.0 or complexity["edge_crossings"] >= 8):
        return "STRUCTURE_TEXTURE", complexity
    return "AI_LOCAL", complexity


def hybrid_inpaint_watermark(rgb: np.ndarray, mask: np.ndarray, engine: str = "AUTO", runtime: LocalModelRuntime | None = None) -> tuple[np.ndarray, tuple[int, int, int, int], dict[str, Any]]:
    """Repair a staged mask while keeping every outside pixel bit-exact."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Ảnh watermark phải là RGB")
    if mask.shape != rgb.shape[:2]:
        raise ValueError("Mask watermark không khớp kích thước ảnh")
    hard_mask = _mask_u8(mask)
    if not np.any(hard_mask):
        raise ValueError("Chưa có vùng watermark để xóa")
    requested = str(engine).upper()
    if requested not in {"AUTO", "FAST", "STRUCTURE_TEXTURE", "AI_LOCAL"}:
        raise ValueError("Engine lấp watermark không hợp lệ")
    selected, complexity = _choose_engine(rgb, hard_mask) if requested == "AUTO" else (requested, _gradient_complexity(rgb, hard_mask))
    started = time.perf_counter()
    diagnostics: dict[str, Any] = {"requested_engine": requested, "selected_engine": selected, "complexity": complexity, "fallback_reason": None}
    repaired: np.ndarray | None = None
    if selected == "AI_LOCAL" and runtime is not None:
        candidate, model_info = runtime.inpaint_rgb(rgb, hard_mask)
        diagnostics["ai_runtime"] = model_info
        if candidate is not None:
            repaired = candidate
        else:
            diagnostics["fallback_reason"], selected = model_info.get("status", "ai_unavailable"), "STRUCTURE_TEXTURE"
    elif selected == "AI_LOCAL":
        diagnostics["fallback_reason"], selected = "ai_runtime_unavailable", "STRUCTURE_TEXTURE"
    if repaired is None and selected == "STRUCTURE_TEXTURE":
        repaired, diagnostics["classical"] = _structure_texture(rgb, hard_mask)
        diagnostics["selected_engine"] = "STRUCTURE_TEXTURE"
    elif repaired is None:
        repaired, diagnostics["selected_engine"], diagnostics["classical"] = _telea(rgb, hard_mask), "FAST", {"status": "ok", "method": "opencv-telea"}
    result, selected_pixels = rgb.copy(), hard_mask > 0
    result[selected_pixels] = repaired[selected_pixels]
    diagnostics.update({"algorithm_version": "watermark-hybrid-v2", "latency_ms": round((time.perf_counter() - started) * 1000.0, 3), "bounds": list(_bounds(selected_pixels)), "changed_pixels": int(np.count_nonzero(np.any(result != rgb, axis=2)))})
    return np.ascontiguousarray(result), _bounds(selected_pixels), diagnostics


def inpaint_watermark(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    repaired, bounds, _ = hybrid_inpaint_watermark(rgb, mask, engine="FAST")
    return repaired, bounds
