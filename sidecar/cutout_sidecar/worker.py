from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from PIL import Image

from .image_core import inference_srgb_copy, load_canonical_png
from .model_runtime import LocalModelRuntime
from .processor import artwork_alpha
from .watermark_engine import confidence_to_soft_mask, detect_watermark, restore_watermark


def atomic_save_array(destination: Path, array: np.ndarray) -> None:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npy", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.save(temporary, np.ascontiguousarray(array, dtype=np.float32), allow_pickle=False)
        # Windows chỉ cho fsync ổn định với handle có quyền ghi.
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_rgb(destination: Path, rgb: np.ndarray) -> None:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".png", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), "RGB").save(temporary, format="PNG")
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))


def process_request(request: dict[str, Any]) -> dict[str, Any]:
    method = request.get("method")
    params = request.get("params") or {}
    if method == "analyze_watermark":
        image_path = Path(params["image_path"]).expanduser().resolve()
        output_path = Path(params["output_path"]).expanduser().resolve()
        rgb = load_rgb_image(image_path)
        detection = detect_watermark(rgb)
        mask = confidence_to_soft_mask(
            detection.confidence,
            feather=float(params.get("feather", 8.0)),
            expand=str(params.get("expand", "MEDIUM")),
        )
        atomic_save_array(output_path, mask)
        return {
            "output_path": str(output_path),
            "shape": list(mask.shape),
            "diagnostics": detection.diagnostics,
            "worker_pid": os.getpid(),
        }
    if method == "restore_watermark":
        image_path = Path(params["image_path"]).expanduser().resolve()
        mask_path = Path(params["mask_path"]).expanduser().resolve()
        output_path = Path(params["output_path"]).expanduser().resolve()
        rgb = load_rgb_image(image_path)
        mask = np.load(mask_path, allow_pickle=False).astype(np.float32)
        runtime = LocalModelRuntime(
            Path(params.get("models_dir") or image_path.parents[3] / "models").resolve()
        )
        repaired, bounds, diagnostics = restore_watermark(
            rgb,
            mask,
            quality=str(params.get("quality", "BALANCED")),
            runtime=runtime,
        )
        atomic_save_rgb(output_path, repaired)
        return {
            "output_path": str(output_path),
            "bounds": list(bounds),
            "diagnostics": diagnostics,
            "worker_pid": os.getpid(),
        }
    if method != "process_artwork":
        raise ValueError(f"Worker method không hỗ trợ: {method}")
    canonical_path = Path(params["canonical_path"]).expanduser().resolve()
    output_path = Path(params["output_path"]).expanduser().resolve()
    if output_path.suffix.lower() != ".npy":
        raise ValueError("Worker output phải là file .npy trong project staging")
    canonical_rgb, source_alpha, icc = load_canonical_png(canonical_path)
    working_rgb_path = params.get("working_rgb_path")
    if working_rgb_path:
        rgb = load_rgb_image(Path(working_rgb_path).expanduser().resolve())
        inference_color_converted = False
    else:
        rgb, inference_color_converted = inference_srgb_copy(canonical_rgb, icc)
    engine_profile = str(params.get("engine_profile", "V3_BALANCED")).upper()
    foreground_points = params.get("foreground_points") or []
    background_points = params.get("background_points") or []
    selection_points = params.get("subject_selection_points")
    quality_preset = str(params.get("quality_preset", "QUALITY")).upper()
    subject_policy = str(params.get("subject_policy", "ALL_DETECTED")).upper()
    source_alpha_mode = str(params.get("source_alpha_mode", "PRESERVE")).upper()
    if not isinstance(foreground_points, list) or not isinstance(background_points, list):
        raise ValueError("foreground_points/background_points phải là danh sách")
    if selection_points is not None and not isinstance(selection_points, list):
        raise ValueError("subject_selection_points phải là danh sách hoặc null")
    runtime = LocalModelRuntime(
        Path(params.get("models_dir") or canonical_path.parents[3] / "models").resolve()
    )
    semantic_alpha = None
    semantic_diagnostics: dict[str, Any] = {"status": "not_requested"}
    topology_diagnostics: dict[str, Any] = {"status": "not_requested"}
    if engine_profile == "V3_AI_LOCAL":
        semantic_alpha, semantic_diagnostics = runtime.semantic_proposal(
            rgb,
            foreground_points,
            auto_detail=quality_preset == "QUALITY",
        )
        if semantic_alpha is not None:
            topology_alpha, topology_diagnostics = runtime.topology_proposal(
                rgb,
                semantic_alpha,
                foreground_points,
                background_points,
            )
            if topology_alpha is not None:
                # SAM2 chỉ quyết định membership/topology; không thay thế alpha fractional.
                topology_threshold = float(params.get("topology_threshold", 0.5))
                if not np.isfinite(topology_threshold):
                    topology_threshold = 0.5
                topology_mask = topology_alpha >= np.clip(topology_threshold, 0.05, 0.95)
                gated_semantic = np.where(topology_mask, semantic_alpha, 0.0).astype(np.float32)
                if foreground_points:
                    # Click bảo vệ có ưu tiên cao hơn topology export chưa đủ parity.
                    topology_diagnostics["applied"] = False
                    topology_diagnostics["status"] = "prompt_preserves_semantic"
                elif np.any(gated_semantic >= 0.5):
                    semantic_alpha = gated_semantic
                    topology_diagnostics["applied"] = True
                else:
                    # Không để một mask topology rỗng xóa sạch proposal đáng tin cậy.
                    topology_diagnostics["status"] = "degenerate_membership"
                    topology_diagnostics["applied"] = False
    alpha, diagnostics, matte_guidance = artwork_alpha(
        rgb,
        source_alpha,
        tolerance=float(params.get("tolerance", 30.0)),
        softness=float(params.get("softness", 18.0)),
        quality_preset=quality_preset,
        engine_profile=engine_profile,
        semantic_alpha=semantic_alpha,
        foreground_points=foreground_points,
        background_points=background_points,
        selection_points=selection_points,
        protection_mode=str(params.get("protection_mode", "CONSERVATIVE")),
        shadow_policy=str(params.get("shadow_policy", "REMOVE")),
        subject_policy=subject_policy,
        source_alpha_mode=source_alpha_mode,
        return_guidance=True,
    )
    matting_diagnostics: dict[str, Any] = {"status": "not_requested"}
    if engine_profile == "V3_AI_LOCAL" and semantic_alpha is not None:
        alpha, matting_diagnostics = runtime.refine_unknown(
            rgb,
            alpha,
            matte_guidance.get("source_alpha_ceiling", source_alpha) if matte_guidance else source_alpha,
            sure_foreground=matte_guidance["sure_foreground"] if matte_guidance else None,
            sure_background=matte_guidance["sure_background"] if matte_guidance else None,
            unknown=matte_guidance["unknown"] if matte_guidance else None,
        )
    ai_models_used = [
        item["model_id"]
        for item in (semantic_diagnostics, topology_diagnostics, matting_diagnostics)
        if item.get("status") == "ok" and item.get("model_id")
    ]
    diagnostics["inference_srgb_copy"] = bool(inference_color_converted)
    diagnostics["ai_runtime"] = {
        "semantic": semantic_diagnostics,
        "topology": topology_diagnostics,
        "matting": matting_diagnostics,
    }
    diagnostics["ai_models_used"] = ai_models_used
    if engine_profile == "V3_AI_LOCAL":
        matting_status = str(matting_diagnostics.get("status", "unavailable"))
        matting_completed = matting_status in {"ok", "no_unknown_roi"}
        if semantic_alpha is None:
            diagnostics["ai_pipeline_status"] = "fallback"
            diagnostics["fallback_reason"] = semantic_diagnostics.get("status", "ai_unavailable")
            diagnostics["selected_strategy"] += "+ai_fallback"
        elif not matting_completed:
            # Proposal đơn lẻ gần như mask nhị phân; lỗi matte phải được báo rõ.
            diagnostics["ai_pipeline_status"] = "degraded"
            diagnostics["needs_review"] = True
            diagnostics["needs_protection"] = True
            diagnostics["result_status"] = "NEEDS_PROTECTION"
            diagnostics["fallback_reason"] = (
                f"matting_{matting_diagnostics.get('status', 'unavailable')}"
            )
        else:
            diagnostics["ai_pipeline_status"] = "complete"
            required_stages = [semantic_diagnostics]
            if topology_diagnostics.get("status") == "ok" and topology_diagnostics.get("applied"):
                required_stages.append(topology_diagnostics)
            if matting_status in {"ok", "no_unknown_roi"}:
                required_stages.append(matting_diagnostics)
            diagnostics["ai_quality_status"] = (
                "qualified"
                if all(bool(stage.get("quality_qualified")) for stage in required_stages)
                else "pending"
            )
    if not np.all(np.isfinite(alpha)):
        raise ValueError("Worker tạo alpha NaN/Inf")
    atomic_save_array(output_path, alpha)
    return {
        "output_path": str(output_path),
        "shape": list(alpha.shape),
        "dtype": "float32",
        "diagnostics": diagnostics,
        "worker_pid": os.getpid(),
    }


def serve_worker_stdio(
    input_stream: TextIO | None = None, output_stream: TextIO | None = None
) -> None:
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    for line in input_stream:
        if not line.strip():
            continue
        request_id: Any = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            result = process_request(request)
            response = {"id": request_id, "ok": True, "result": result}
        except Exception as error:
            response = {
                "id": request_id,
                "ok": False,
                "error": {
                    "code": type(error).__name__.upper(),
                    "message": str(error),
                    "details": traceback.format_exc(limit=8),
                },
            }
        output_stream.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
        output_stream.flush()


if __name__ == "__main__":
    serve_worker_stdio()
