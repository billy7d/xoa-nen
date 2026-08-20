#!/usr/bin/env python3
"""Đo alpha và mức giữ chi tiết nguồn trong các ROI của ảnh xóa nền."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import numpy as np
from PIL import Image


def _load_source(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_candidate(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if path.suffix.lower() == ".npy":
        alpha = np.load(path, allow_pickle=False).astype(np.float32)
        return np.clip(alpha, 0.0, 1.0), None
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return rgba[..., 3].astype(np.float32) / 255.0, rgba[..., :3]


def _parse_roi(value: str, width: int, height: int) -> tuple[str, tuple[int, int, int, int]]:
    try:
        name, coordinates = value.split(":", 1)
        x0, y0, x1, y1 = (int(part) for part in coordinates.split(","))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "ROI phải có dạng ten:x0,y0,x1,y1"
        ) from error
    if not name or not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise argparse.ArgumentTypeError(f"ROI ngoài ảnh hoặc rỗng: {value}")
    return name, (x0, y0, x1, y1)


def _alpha_metrics(alpha: np.ndarray, gradient: np.ndarray) -> dict[str, Any]:
    alpha_u8 = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    gradient_sum = float(np.sum(gradient))
    gradient_coverage = (
        100.0 * float(np.sum(gradient * alpha)) / gradient_sum if gradient_sum > 0 else 100.0
    )
    threshold = float(np.percentile(gradient, 75.0))
    strong = gradient >= threshold
    strong_count = int(np.count_nonzero(strong))
    return {
        "pixel_count": int(alpha.size),
        "alpha_mean": round(float(np.mean(alpha)), 8),
        "alpha_equivalent_px": round(float(np.sum(alpha)), 4),
        "alpha_zero_pct": round(100.0 * float(np.mean(alpha_u8 == 0)), 6),
        "alpha_lt16_pct": round(100.0 * float(np.mean(alpha_u8 < 16)), 6),
        "alpha_16_239_pct": round(
            100.0 * float(np.mean((alpha_u8 >= 16) & (alpha_u8 <= 239))), 6
        ),
        "alpha_ge128_px": int(np.count_nonzero(alpha_u8 >= 128)),
        "alpha_ge240_pct": round(100.0 * float(np.mean(alpha_u8 >= 240)), 6),
        "alpha_255_pct": round(100.0 * float(np.mean(alpha_u8 == 255)), 6),
        "alpha_max": int(np.max(alpha_u8)),
        "gradient_weighted_alpha_coverage_pct": round(gradient_coverage, 6),
        "top25_gradient_suppressed_pct": round(
            100.0 * float(np.count_nonzero(strong & (alpha_u8 < 16))) / max(1, strong_count),
            6,
        ),
    }


def _measure(
    source: np.ndarray,
    alpha: np.ndarray,
    candidate_rgb: np.ndarray | None,
    rois: list[tuple[str, tuple[int, int, int, int]]],
) -> dict[str, Any]:
    if alpha.shape != source.shape[:2]:
        raise ValueError(f"Kích thước alpha {alpha.shape} khác ảnh nguồn {source.shape[:2]}")
    if not np.all(np.isfinite(alpha)):
        raise ValueError("Alpha chứa NaN hoặc Inf")
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gx, gy)
    report: dict[str, Any] = {
        "width": int(source.shape[1]),
        "height": int(source.shape[0]),
        "alpha_finite": True,
        "alpha_min": round(float(np.min(alpha)), 8),
        "alpha_max": round(float(np.max(alpha)), 8),
        "rgb_exact_source": (
            bool(np.array_equal(candidate_rgb, source)) if candidate_rgb is not None else None
        ),
        "rois": {},
    }
    for name, (x0, y0, x1, y1) in rois:
        report["rois"][name] = {
            "bounds": [x0, y0, x1, y1],
            **_alpha_metrics(alpha[y0:y1, x0:x1], gradient[y0:y1, x0:x1]),
        }
    return report


def _ground_truth_metrics(candidate: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    if candidate.shape != truth.shape:
        raise ValueError("Ground-truth alpha khác kích thước candidate")
    difference = np.abs(candidate - truth)
    candidate_gradient_x = cv2.Sobel(candidate, cv2.CV_32F, 1, 0, ksize=3)
    candidate_gradient_y = cv2.Sobel(candidate, cv2.CV_32F, 0, 1, ksize=3)
    truth_gradient_x = cv2.Sobel(truth, cv2.CV_32F, 1, 0, ksize=3)
    truth_gradient_y = cv2.Sobel(truth, cv2.CV_32F, 0, 1, ksize=3)
    candidate_binary = candidate >= 0.5
    truth_binary = truth >= 0.5
    intersection = int(np.count_nonzero(candidate_binary & truth_binary))
    union = int(np.count_nonzero(candidate_binary | truth_binary))
    return {
        "sad_raw": round(float(np.sum(difference)), 6),
        "sad_per_megapixel": round(float(np.sum(difference)) * 1_000_000.0 / candidate.size, 6),
        "mse": round(float(np.mean(np.square(candidate - truth))), 10),
        "gradient_l1_mean": round(
            float(
                np.mean(np.abs(candidate_gradient_x - truth_gradient_x))
                + np.mean(np.abs(candidate_gradient_y - truth_gradient_y))
            ),
            10,
        ),
        "binary_iou_0_5": round(intersection / max(1, union), 8),
        "background_leakage_mean": round(
            float(np.mean(candidate[truth <= 0.01])) if np.any(truth <= 0.01) else 0.0,
            10,
        ),
        "opaque_miss_mean": round(
            float(np.mean(1.0 - candidate[truth >= 0.99])) if np.any(truth >= 0.99) else 0.0,
            10,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Ảnh RGB/RGBA nguồn")
    parser.add_argument("--candidate", type=Path, required=True, help="PNG RGBA hoặc alpha NPY")
    parser.add_argument("--baseline", type=Path, help="PNG/NPY cũ để so sánh")
    parser.add_argument("--ground-truth", type=Path, help="Matte GT PNG/NPY nếu có")
    parser.add_argument(
        "--roi",
        action="append",
        default=[],
        help="ROI lặp lại dạng ten:x0,y0,x1,y1; mặc định đo toàn ảnh",
    )
    parser.add_argument("--output", type=Path, help="Ghi JSON; nếu bỏ trống sẽ in stdout")
    args = parser.parse_args()

    source = _load_source(args.source.resolve())
    candidate_alpha, candidate_rgb = _load_candidate(args.candidate.resolve())
    parsed_rois = [
        _parse_roi(value, source.shape[1], source.shape[0]) for value in args.roi
    ] or [("full", (0, 0, source.shape[1], source.shape[0]))]
    report: dict[str, Any] = {
        "source": str(args.source.resolve()),
        "candidate": str(args.candidate.resolve()),
        "metric_note": (
            "gradient_weighted_alpha_coverage chỉ là proxy coverage trong ROI; "
            "alpha=1 toàn ROI sẽ tối đa hóa nó nên không được dùng thay ground-truth."
        ),
        "candidate_metrics": _measure(source, candidate_alpha, candidate_rgb, parsed_rois),
    }
    if args.baseline:
        baseline_alpha, baseline_rgb = _load_candidate(args.baseline.resolve())
        report["baseline"] = str(args.baseline.resolve())
        report["baseline_metrics"] = _measure(
            source, baseline_alpha, baseline_rgb, parsed_rois
        )
        report["alpha_mae_vs_baseline"] = round(
            float(np.mean(np.abs(candidate_alpha - baseline_alpha))), 8
        )
    if args.ground_truth:
        truth_alpha, _ = _load_candidate(args.ground_truth.resolve())
        report["ground_truth"] = str(args.ground_truth.resolve())
        report["ground_truth_metrics"] = _ground_truth_metrics(candidate_alpha, truth_alpha)

    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        destination = args.output.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
