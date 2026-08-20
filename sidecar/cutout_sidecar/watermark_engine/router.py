from __future__ import annotations

from typing import Any


def choose_route(
    analysis: dict[str, Any],
    quality: str = "BALANCED",
    ai_fast_available: bool = False,
    ai_quality_available: bool = False,
) -> dict[str, Any]:
    normalized_quality = str(quality).upper()
    if normalized_quality == "MAX":
        normalized_quality = "MAXIMUM"
    mask_ratio = float(analysis.get("mask_ratio", 0.0))
    texture = str(analysis.get("texture", "TEXTURE")).upper()
    transparency = float(analysis.get("transparency_score", 0.0))
    structure = float(analysis.get("structure_score", 0.0))
    semantic = float(analysis.get("semantic_complexity", 0.0))
    if transparency >= 0.62 and mask_ratio <= 0.22:
        primary = "DEBLEND"
    elif texture in {"GEOMETRIC", "TEXTURE"} and structure >= 0.18 and mask_ratio <= 0.18:
        primary = "PATCH_RESTORE"
    elif mask_ratio <= 0.0035 and texture in {"FLAT", "GRADIENT"}:
        primary = "TELEA_FAST"
    elif normalized_quality == "MAXIMUM" and ai_quality_available:
        primary = "AI_QUALITY"
    elif ai_fast_available and normalized_quality in {"BALANCED", "MAXIMUM"}:
        primary = "AI_FAST"
    elif semantic >= 0.45 and texture != "FLAT":
        primary = "PATCH_RESTORE"
    else:
        primary = "TELEA_FAST"

    fallbacks = [primary]
    for route in (
        "PATCH_RESTORE",
        "AI_FAST" if ai_fast_available else "",
        "AI_QUALITY" if ai_quality_available and normalized_quality == "MAXIMUM" else "",
        "TELEA_FAST",
    ):
        if route and route not in fallbacks:
            fallbacks.append(route)
    return {
        "primary": primary,
        "fallbacks": fallbacks[:3],
        "quality": normalized_quality,
        "ai_fast_available": bool(ai_fast_available),
        "ai_quality_available": bool(ai_quality_available),
    }
