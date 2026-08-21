from __future__ import annotations

from typing import Any

import numpy as np

from .inpaint import ai_restore
from .mask import bounds_from_mask
from .quality import score_candidate
from .router import choose_route


def _ai_role(route: str) -> str:
    return "watermark_inpaint_quality" if route == "AI_QUALITY" else "watermark_inpaint_fast"


def _route_candidate(
    route: str,
    rgb: np.ndarray,
    soft_mask: np.ndarray,
    quality: str,
    runtime: Any | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
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
    ai_fast_available = bool(runtime is not None and getattr(runtime, "has_role", lambda _role: False)("watermark_inpaint_fast"))
    ai_quality_available = bool(runtime is not None and getattr(runtime, "has_role", lambda _role: False)("watermark_inpaint_quality"))
    routing = choose_route(quality, ai_fast_available, ai_quality_available)
    if not routing["fallbacks"]:
        raise RuntimeError(
            "Cần model AI local lấp nền watermark. Hãy cài model-pack ONNX có role "
            "watermark_inpaint_fast hoặc watermark_inpaint_quality."
        )
    minimum_score = {
        "FAST": 0.74,
        "BALANCED": 0.80,
        "MAXIMUM": 0.84,
    }.get(str(routing["quality"]), 0.80)
    minimum_change = 2.0
    accepted: tuple[np.ndarray, dict[str, Any]] | None = None
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
        score = score_candidate(rgb, candidate, mask)
        diagnostics = {
            **diagnostics,
            "status": "ok",
            "quality": score,
            "minimum_quality": minimum_score,
        }
        quality_passed = (
            float(score.get("overall", 0.0)) >= minimum_score
            and float(score.get("changed_mean", 0.0)) >= minimum_change
        )
        if quality_passed:
            accepted = (candidate, diagnostics)
            attempts.append(
                {
                    "route": route,
                    "status": "ok",
                    "model_id": diagnostics.get("model_id"),
                    "overall": score.get("overall"),
                }
            )
            # Một model AI vượt quality gate là kết quả cuối; không hậu xử lý bằng thuật toán.
            break
        attempts.append(
            {
                "route": route,
                "status": "quality_rejected",
                "model_id": diagnostics.get("model_id"),
                "overall": score.get("overall"),
                "minimum_quality": minimum_score,
                "minimum_change": minimum_change,
            }
        )
    if accepted is None:
        summary = "; ".join(
            f"{item.get('route')}: {item.get('status', 'failed')}"
            + (f" ({item.get('overall')})" if item.get("overall") is not None else "")
            for item in attempts
        )
        raise RuntimeError(
            "Model AI local chưa tái tạo nền đạt quality gate; ảnh chưa bị thay đổi. "
            "Hãy chỉnh mask phủ kín watermark hoặc chọn mức chất lượng khác. "
            f"Chi tiết: {summary or 'không có kết quả inference'}"
        )
    best = accepted
    bounds = bounds_from_mask(mask, 0.005)
    diagnostics = {
        "algorithm_version": "watermark-restore-v3.1-ai-local-gated",
        "routing": routing,
        "attempts": attempts,
        "selected": best[1],
        "quality_gate": {
            "status": "PASS",
            "minimum": minimum_score,
            "overall": best[1].get("quality", {}).get("overall"),
            "minimum_change": minimum_change,
            "changed_mean": best[1].get("quality", {}).get("changed_mean"),
        },
        "bounds": list(bounds),
        "pixel_preservation": "outside_bounds_unchanged",
    }
    return np.ascontiguousarray(best[0], dtype=np.uint8), bounds, diagnostics
