from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from .processor import analyze_components


def run_preflight(
    alpha: np.ndarray,
    width: int,
    height: int,
    print_width_inch: float | None = None,
    print_height_inch: float | None = None,
    color_profile: str = "sRGB",
    source_converted: bool = False,
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if not np.all(np.isfinite(alpha)):
        failures.append({"code": "ALPHA_INVALID", "message": "Alpha chứa NaN hoặc Inf"})

    clipped = {
        "top": bool(np.any(alpha[0] > 0.01)),
        "bottom": bool(np.any(alpha[-1] > 0.01)),
        "left": bool(np.any(alpha[:, 0] > 0.01)),
        "right": bool(np.any(alpha[:, -1] > 0.01)),
    }
    if any(clipped.values()):
        warnings.append(
            {
                "code": "SUBJECT_TOUCHES_CANVAS",
                "message": "Foreground chạm biên canvas; hãy kiểm tra clipping hoặc padding.",
                "details": clipped,
            }
        )

    residue_count = int(np.count_nonzero((alpha > 0.0) & (alpha <= 0.02)))
    semi_count = int(np.count_nonzero((alpha > 0.02) & (alpha < 0.98)))
    opaque_count = int(np.count_nonzero(alpha >= 0.98))
    total = int(alpha.size)

    if residue_count:
        warnings.append(
            {
                "code": "LOW_ALPHA_RESIDUE",
                "message": f"Có {residue_count:,} pixel alpha <= 2%; chúng vẫn có thể tạo residue khi in.",
            }
        )
    if semi_count / max(1, total) > 0.01:
        warnings.append(
            {
                "code": "SEMI_TRANSPARENCY",
                "message": "Artwork có vùng bán trong suốt đáng kể; hãy preview trên màu garment mục tiêu.",
            }
        )
    if opaque_count / max(1, total) > 0.99:
        warnings.append(
            {
                "code": "OPAQUE_RECTANGLE",
                "message": "Gần như toàn bộ canvas opaque; có thể background chưa được xóa.",
            }
        )

    components = analyze_components(alpha)
    small_components = [component for component in components if component["area_px"] < 64]
    if small_components:
        warnings.append(
            {
                "code": "SMALL_COMPONENTS_NEED_REVIEW",
                "message": f"Có {len(small_components)} component nhỏ. App không tự xóa để tránh mất grunge/chi tiết có chủ ý.",
            }
        )

    effective_ppi = None
    if print_width_inch and print_height_inch:
        if print_width_inch <= 0 or print_height_inch <= 0:
            failures.append(
                {"code": "PRINT_SIZE_INVALID", "message": "Kích thước in phải lớn hơn 0"}
            )
        else:
            effective_ppi = {
                "x": width / float(print_width_inch),
                "y": height / float(print_height_inch),
            }
            if min(effective_ppi.values()) < 150:
                warnings.append(
                    {
                        "code": "LOW_EFFECTIVE_PPI",
                        "message": "Effective PPI dưới 150; app không tự upscale.",
                        "details": effective_ppi,
                    }
                )

    if color_profile.lower() != "srgb":
        warnings.append(
            {
                "code": "PROFILE_NOT_SRGB",
                "message": "POD-ready nên được color-convert và embed sRGB.",
            }
        )
    if source_converted:
        warnings.append(
            {
                "code": "SOURCE_CONVERTED",
                "message": "Source đã được chuyển color mode/profile khi canonical decode.",
            }
        )

    if failures:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "report_version": "preflight-v1",
        "status": status,
        "effective_ppi": effective_ppi,
        "pixel_dimensions": {"width": width, "height": height},
        "alpha_statistics": {
            "min": float(np.min(alpha)),
            "max": float(np.max(alpha)),
            "mean": float(np.mean(alpha)),
            "transparent_pixels": int(np.count_nonzero(alpha <= 0.001)),
            "opaque_pixels": opaque_count,
            "semi_transparent_pixels": semi_count,
            "low_alpha_residue_pixels": residue_count,
        },
        "component_statistics": {
            "count": len(components),
            "small_count": len(small_components),
            "components": components[:100],
        },
        "color_profile": color_profile,
        "warnings": warnings,
        "failures": failures,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

