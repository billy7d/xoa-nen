from __future__ import annotations

from typing import Any


def choose_route(
    quality: str = "BALANCED",
    ai_fast_available: bool = False,
    ai_quality_available: bool = False,
) -> dict[str, Any]:
    normalized_quality = str(quality).upper()
    if normalized_quality == "MAX":
        normalized_quality = "MAXIMUM"
    if normalized_quality == "MAXIMUM" and ai_quality_available:
        primary = "AI_QUALITY"
    elif ai_fast_available:
        primary = "AI_FAST"
    elif ai_quality_available:
        # Người dùng vẫn có thể chạy model quality khi chỉ có pack đó được cài.
        primary = "AI_QUALITY"
    else:
        primary = None

    # Chỉ thử các model AI local khác khi model ưu tiên không chạy; không có fallback thuật toán.
    fallbacks = [primary] if primary else []
    for route in ("AI_QUALITY" if ai_quality_available else "", "AI_FAST" if ai_fast_available else ""):
        if route and route not in fallbacks:
            fallbacks.append(route)
    return {
        "primary": primary,
        "fallbacks": fallbacks,
        "quality": normalized_quality,
        "ai_fast_available": bool(ai_fast_available),
        "ai_quality_available": bool(ai_quality_available),
        "requires_local_ai": True,
    }
