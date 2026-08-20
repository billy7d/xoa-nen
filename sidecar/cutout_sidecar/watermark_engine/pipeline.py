from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .analyzer import analyze_watermark
from .deblend import attenuate_transparent_overlay
from .harmonize import harmonize_boundary
from .inpaint import ai_restore, expanded_bounds, restore_roi_with_candidate, telea_restore
from .mask import bounds_from_mask, hard_mask
from .patch_restore import reference_patch_restore
from .quality import score_candidate
from .router import choose_route


def _candidate_patch_restore(
    rgb: np.ndarray,
    soft_mask: np.ndarray,
    quality: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = soft_mask.shape
    roi = expanded_bounds(soft_mask, width, height, quality)
    if roi == (0, 0, 0, 0):
        raise ValueError("Chưa có vùng watermark để xóa")
    x0, y0, x1, y1 = roi
    radius = 2 if str(quality).upper() == "FAST" else 3
    if str(quality).upper() == "MAXIMUM":
        radius = 4
    candidate_roi = reference_patch_restore(
        rgb[y0:y1, x0:x1],
        hard_mask(soft_mask[y0:y1, x0:x1]),
        radius=radius,
    )
    return restore_roi_with_candidate(rgb, soft_mask, candidate_roi, roi), {
        "route": "PATCH_RESTORE",
        "roi": list(roi),
        "radius": radius,
    }


def _candidate_deblend(
    rgb: np.ndarray,
    soft_mask: np.ndarray,
    quality: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    deb = attenuate_transparent_overlay(rgb, soft_mask)
    # Deblend xử lý watermark mờ; phần core còn sót được lấp rất nhẹ trong ROI.
    residue_mask = np.where(soft_mask >= 0.72, soft_mask, 0.0).astype(np.float32)
    if np.any(residue_mask > 0.01):
        telea, telea_diag = telea_restore(deb, residue_mask, quality)
        deb = telea
        telea_route = telea_diag
    else:
        telea_route = None
    return deb, {"route": "DEBLEND", "residue_inpaint": telea_route}


def _ai_role(route: str) -> str:
    return "watermark_inpaint_quality" if route == "AI_QUALITY" else "watermark_inpaint_fast"


def _route_candidate(
    route: str,
    rgb: np.ndarray,
    soft_mask: np.ndarray,
    quality: str,
    runtime: Any | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if route == "DEBLEND":
        return _candidate_deblend(rgb, soft_mask, quality)
    if route == "PATCH_RESTORE":
        return _candidate_patch_restore(rgb, soft_mask, quality)
    if route == "TELEA_FAST":
        return telea_restore(rgb, soft_mask, quality)
    if route in {"AI_FAST", "AI_QUALITY"}:
        return ai_restore(rgb, soft_mask, runtime, quality, _ai_role(route))
    return None, {"status": "unsupported_route", "route": route}


def restore_watermark(
    rgb: np.ndarray,
    soft_mask: np.ndarray,
    quality: str = "BALANCED",
    runtime: Any | None = None,
    max_retries: int = 2,
) -> tuple[np.ndarray, tuple[int, int, int, int], dict[str, Any]]:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Ảnh watermark phải là RGB")
    mask = np.clip(np.asarray(soft_mask, dtype=np.float32), 0.0, 1.0)
    if mask.shape != rgb.shape[:2]:
        raise ValueError("Mask watermark không khớp kích thước ảnh")
    if not np.any(mask > 0.01):
        raise ValueError("Chưa có vùng watermark để xóa")
    analysis = analyze_watermark(rgb, mask)
    ai_fast_available = bool(runtime is not None and getattr(runtime, "has_role", lambda _role: False)("watermark_inpaint_fast"))
    ai_quality_available = bool(runtime is not None and getattr(runtime, "has_role", lambda _role: False)("watermark_inpaint_quality"))
    routing = choose_route(analysis, quality, ai_fast_available, ai_quality_available)
    candidates: list[tuple[np.ndarray, dict[str, Any], dict[str, Any]]] = []
    attempts: list[dict[str, Any]] = []
    for route in routing["fallbacks"][: max(1, max_retries + 1)]:
        try:
            candidate, diagnostics = _route_candidate(route, rgb, mask, routing["quality"], runtime)
        except Exception as error:
            attempts.append({"route": route, "status": "failed", "error": f"{type(error).__name__}: {error}"})
            continue
        if candidate is None:
            attempts.append({"route": route, **diagnostics})
            continue
        harmonized = harmonize_boundary(rgb, candidate, mask)
        score = score_candidate(rgb, harmonized, mask)
        diagnostics = {**diagnostics, "status": "ok", "quality": score}
        candidates.append((harmonized, diagnostics, score))
        attempts.append({"route": route, "status": "ok", "overall": score["overall"]})
        if float(score["overall"]) >= 0.86:
            break
    if not candidates:
        fallback, diagnostics = telea_restore(rgb, mask, routing["quality"])
        harmonized = harmonize_boundary(rgb, fallback, mask)
        score = score_candidate(rgb, harmonized, mask)
        candidates.append((harmonized, {**diagnostics, "status": "ok", "quality": score}, score))
    best = max(candidates, key=lambda item: float(item[2].get("overall", 0.0)))
    changed = cv2.dilate((mask > 0.005).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    bounds = bounds_from_mask(changed, 0)
    diagnostics = {
        "algorithm_version": "watermark-restore-v2-router",
        "analysis": analysis,
        "routing": routing,
        "attempts": attempts,
        "selected": best[1],
        "bounds": list(bounds),
        "pixel_preservation": "outside_bounds_unchanged",
    }
    return np.ascontiguousarray(best[0], dtype=np.uint8), bounds, diagnostics
