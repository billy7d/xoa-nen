"""Watermark Removal v2: session mask, detector và restoration router."""
from __future__ import annotations

from .detector import WatermarkDetection, detect_watermark
from .mask import bounds_from_mask, confidence_to_soft_mask, hard_mask, rasterize_stroke
from .pipeline import restore_watermark

__all__ = [
    "WatermarkDetection",
    "bounds_from_mask",
    "confidence_to_soft_mask",
    "detect_watermark",
    "hard_mask",
    "rasterize_stroke",
    "restore_watermark",
]
