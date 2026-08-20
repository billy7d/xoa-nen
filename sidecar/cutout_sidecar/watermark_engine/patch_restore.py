from __future__ import annotations

import cv2
import numpy as np


def reference_patch_restore(rgb: np.ndarray, hard_mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Khôi phục theo hướng cấu trúc bằng Navier-Stokes cho các vùng có đường/texture."""
    mask = (hard_mask > 0).astype(np.uint8) * 255
    if not np.any(mask):
        return rgb.copy()
    return cv2.inpaint(rgb, mask, max(1, int(radius)), cv2.INPAINT_NS)
