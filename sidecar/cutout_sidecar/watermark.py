"""Tiện ích chọn watermark và tái tạo ảnh cục bộ.

Bộ dò được thiết kế bảo thủ: chỉ đề xuất các nét tần số cao nhỏ, giống chữ.
Với artwork không có đặc điểm này, người dùng luôn có thể tô mask thủ công.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def automatic_watermark_mask(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a conservative binary mask for small overlay-like components."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    # Watermark thường mảnh và sáng/tối hơn nền lân cận; kết hợp hai cực để
    # không mặc định watermark là chữ màu trắng.
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(3.0, min(width, height) / 90))
    residual = cv2.absdiff(gray, blur)
    _, candidate = cv2.threshold(residual, 22, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 70, 150)
    candidate = cv2.bitwise_and(candidate, cv2.dilate(edges, np.ones((3, 3), np.uint8)))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    selected = np.zeros_like(candidate)
    image_area = width * height
    for label in range(1, count):
        x, y, component_w, component_h, area = stats[label]
        if area < 8 or area > image_area * 0.035:
            continue
        # Vùng đặc lớn thường là nội dung ảnh, không phải lớp phủ. Nét chữ/logo
        # thường thưa bên trong một khung bao nhỏ gọn.
        fill = area / max(1, component_w * component_h)
        if component_w < 2 or component_h < 2 or fill > 0.72:
            continue
        selected[labels == label] = 255
    # Inpainting cần phủ thêm một ít ở viền chữ đã khử răng cưa.
    selected = cv2.dilate(selected, np.ones((3, 3), np.uint8), iterations=1)
    return selected, {"pixels": int(np.count_nonzero(selected)), "bounds": list(_bounds(selected > 0))}


def brush_mask(shape: tuple[int, int], points: list[dict[str, float]], radius: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not points:
        return mask
    radius_px = max(1, min(int(round(float(radius))), max(shape)))
    previous: tuple[int, int] | None = None
    for point in points:
        center = (int(round(float(point["x"]))), int(round(float(point["y"]))))
        cv2.circle(mask, center, radius_px, 255, thickness=-1, lineType=cv2.LINE_AA)
        if previous is not None:
            cv2.line(mask, previous, center, 255, thickness=radius_px * 2, lineType=cv2.LINE_AA)
        previous = center
    return mask


def inpaint_watermark(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Ảnh watermark phải là RGB")
    if mask.shape != rgb.shape[:2]:
        raise ValueError("Mask watermark không khớp kích thước ảnh")
    if not np.any(mask):
        raise ValueError("Chưa có vùng watermark để xóa")
    # TELEA giữ texture lân cận tốt với chữ mảnh; luôn xử lý ở độ phân giải gốc,
    # không thay đổi kích thước ảnh trong thao tác này.
    repaired = cv2.inpaint(rgb, (mask > 0).astype(np.uint8) * 255, 4, cv2.INPAINT_TELEA)
    return np.ascontiguousarray(repaired), _bounds(mask > 0)
