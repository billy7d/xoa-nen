from __future__ import annotations

import json
import os
import platform
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from . import __version__
from .edits import apply_brush, apply_magic_wand, apply_wand_coverage
from .exports import export_image
from .image_core import (
    decode_canonical,
    inference_srgb_copy,
    load_canonical_png,
    save_preview,
)
from .models import (
    download_model_pack,
    install_model_pack,
    list_model_manifests,
    remove_model_pack,
)
from .preflight import run_preflight
from .processor import analyze_components, magic_wand_selection, select_components
from .project_store import ProjectStore, atomic_write_json
from .worker_supervisor import WorkerSupervisor
from .watermark import automatic_watermark_mask, brush_mask, inpaint_watermark


class Coordinator:
    def __init__(self, store: ProjectStore | None = None) -> None:
        self.store = store or ProjectStore()
        self.worker = WorkerSupervisor()
        self.models_dir = Path(
            os.environ.get("CUTOUT_MODELS_DIR", Path(__file__).resolve().parents[2] / "models")
        ).resolve()

    def dispatch(self, method: str, params: dict[str, Any]) -> Any:
        handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "health": self.health,
            "import_image": self.import_image,
            "get_project": self.get_project,
            "process_artwork": self.process_artwork,
            "remove_watermark": self.remove_watermark,
            "apply_brush": self.brush,
            "apply_magic_wand": self.magic_wand,
            "preview_magic_wand": self.preview_magic_wand,
            "commit_magic_wand": self.commit_magic_wand,
            "cancel_magic_wand": self.cancel_magic_wand,
            "set_subject_selection": self.set_subject_selection,
            "undo": self.undo,
            "redo": self.redo,
            "preflight": self.preflight,
            "export": self.export,
            "list_models": self.list_models,
            "install_model_pack": self.install_model_pack,
            "download_model_pack": self.download_model_pack,
            "remove_model_pack": self.remove_model_pack,
        }
        handler = handlers.get(method)
        if not handler:
            raise ValueError(f"Method không hỗ trợ: {method}")
        return handler(params)

    def health(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "offline": True,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "projects_dir": str(self.store.root),
            "processing_engine": "hybrid-cutout-v3",
        }

    @staticmethod
    def _processing_points(value: Any, width: int, height: int, name: str) -> list[dict[str, float]]:
        """Kiểm tra và cố định prompt trong hệ toạ độ canonical của project."""
        if value is None:
            return []
        if not isinstance(value, list) or len(value) > 16:
            raise ValueError(f"{name} phải là danh sách tối đa 16 điểm")
        points: list[dict[str, float]] = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError(f"{name} có phần tử không hợp lệ")
            try:
                x = float(item["x"])
                y = float(item["y"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{name} có toạ độ không hợp lệ") from error
            if not np.isfinite(x) or not np.isfinite(y):
                raise ValueError(f"{name} có toạ độ không hữu hạn")
            points.append(
                {
                    "x": round(min(width - 0.5, max(0.5, x)), 3),
                    "y": round(min(height - 0.5, max(0.5, y)), 3),
                }
            )
        return points

    def import_image(self, params: dict[str, Any]) -> dict[str, Any]:
        canonical = decode_canonical(params["path"])
        manifest = self.store.create(canonical)
        project_id = manifest["project_id"]
        return self._project_payload(project_id, preview_name="original")

    def get_project(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._project_payload(params["project_id"])

    def process_artwork(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = params["project_id"]
        rgb, _, _ = load_canonical_png(self.store.canonical_path(project_id))
        engine_profile = str(params.get("engine_profile", "V3_BALANCED")).upper()
        subject_policy = str(params.get("subject_policy", "ALL_DETECTED")).upper()
        foreground_points = self._processing_points(
            params.get("foreground_points"), rgb.shape[1], rgb.shape[0], "foreground_points"
        )
        background_points = self._processing_points(
            params.get("background_points"), rgb.shape[1], rgb.shape[0], "background_points"
        )
        protection_mode = str(params.get("protection_mode", "CONSERVATIVE")).upper()
        shadow_policy = str(params.get("shadow_policy", "REMOVE")).upper()
        if subject_policy not in {"ALL_DETECTED", "SELECTED"}:
            raise ValueError("Subject policy không hợp lệ")
        if protection_mode != "CONSERVATIVE" or shadow_policy != "REMOVE":
            raise ValueError("Cấu hình bảo toàn vật thể hoặc bóng không hợp lệ")
        staging_path = self.store.path(project_id) / "alpha" / "staging" / "worker-alpha.npy"
        worker_result = self.worker.request(
            "process_artwork",
            {
                "canonical_path": str(self.store.canonical_path(project_id)),
                "output_path": str(staging_path),
                "tolerance": float(params.get("tolerance", 30.0)),
                "softness": float(params.get("softness", 18.0)),
                "quality_preset": str(params.get("quality_preset", "QUALITY")),
                "engine_profile": engine_profile,
                "models_dir": str(self.models_dir),
                "foreground_points": foreground_points,
                "background_points": background_points,
                "protection_mode": protection_mode,
                "shadow_policy": shadow_policy,
            },
        )
        try:
            alpha = np.load(staging_path, allow_pickle=False)
        finally:
            staging_path.unlink(missing_ok=True)
        diagnostics = worker_result["diagnostics"]
        diagnostics["worker_pid"] = worker_result["worker_pid"]
        if not np.all(np.isfinite(alpha)):
            raise ValueError("Processing tạo alpha NaN/Inf")
        self.store.write_alpha(project_id, "base", alpha)
        self.store.write_alpha(project_id, "current", alpha)
        self.store.reset_history(project_id)
        subjects = analyze_components(alpha)
        for subject in subjects:
            subject["selected"] = True
            subject["confidence"] = "review" if subject["needs_review"] else "detected"
        warnings: list[dict[str, Any]] = []
        if engine_profile == "V3_AI_LOCAL" and not diagnostics.get("ai_models_used"):
            warnings.append(
                {
                    "code": "AI_LOCAL_FALLBACK",
                    "message": "Gói AI chưa sẵn sàng; đã fallback an toàn sang V3 Cân bằng.",
                }
            )
        if diagnostics.get("needs_protection"):
            warnings.append(
                {
                    "code": "NEEDS_PROTECTION",
                    "message": "Ảnh có candidate mâu thuẫn. Chọn Khóa vật thể (P), bấm vào thân và chạy lại để bảo toàn vật thể.",
                }
            )
        elif diagnostics.get("needs_review"):
            warnings.append(
                {
                    "code": "CUTOUT_NEEDS_REVIEW",
                    "message": "Các candidate chưa đồng thuận; hãy kiểm tra vùng màu vàng hoặc chọn vật thể.",
                }
            )
        manifest = self.store.manifest(project_id)
        manifest["processing"] = {
            "content_mode": "AUTO",
            "engine_profile": engine_profile,
            "quality_preset": diagnostics["quality_preset"],
            "subject_policy": subject_policy,
            "foreground_points": foreground_points,
            "background_points": background_points,
            "protection_mode": protection_mode,
            "shadow_policy": shadow_policy,
            "result_status": diagnostics.get("result_status", "READY"),
            "diagnostics": diagnostics,
            "ai_models_used": diagnostics.get("ai_models_used", []),
            "subjects": subjects[:100],
            "selected_subject_ids": [int(subject["id"]) for subject in subjects],
            "review_regions": diagnostics.get("review_regions", []),
            "warnings": warnings,
        }
        self.store.update_manifest(project_id, manifest)
        self._refresh_preview(project_id, self.store.read_working_rgb(project_id), alpha)
        return self._project_payload(project_id)

    def remove_watermark(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = params["project_id"]
        mode = str(params.get("mode", "AUTO")).upper()
        rgb = self.store.read_working_rgb(project_id)
        if mode == "AUTO":
            mask, detection = automatic_watermark_mask(rgb)
            operation = {"tool": "watermark_auto", "algorithm_version": "opencv-telea-v1", "detection": detection}
        elif mode == "MANUAL":
            points = self._processing_points(params.get("points"), rgb.shape[1], rgb.shape[0], "points")
            mask = brush_mask(rgb.shape[:2], points, float(params.get("radius", 24)))
            detection = {"pixels": int(np.count_nonzero(mask)), "bounds": []}
            operation = {"tool": "watermark_manual", "algorithm_version": "opencv-telea-v1", "pixels": detection["pixels"]}
        else:
            raise ValueError("Chế độ xoá watermark không hợp lệ")
        repaired, bounds = inpaint_watermark(rgb, mask)
        operation["bounds"] = list(bounds)
        self.store.commit_retouch_edit(project_id, rgb, repaired, bounds, operation)
        alpha = self.store.read_alpha(project_id)
        self._refresh_preview(project_id, repaired, alpha)
        return self._project_payload(project_id)

    def brush(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = params["project_id"]
        rgb, source_alpha, _ = load_canonical_png(self.store.canonical_path(project_id))
        before = self.store.read_alpha(project_id)
        locks = self.store.read_lock(project_id)
        after, bounds, operation = apply_brush(
            before,
            source_alpha,
            locks,
            params["points"],
            params.get("radius", 16),
            params.get("hardness", 0.8),
            params.get("opacity", 1.0),
            params.get("mode", "remove"),
            params.get("target_alpha", 1.0),
        )
        self.store.commit_alpha_edit(project_id, before, after, bounds, operation)
        self._refresh_preview(project_id, self.store.read_working_rgb(project_id), after)
        return self._project_payload(project_id)

    def magic_wand(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = params["project_id"]
        canonical_rgb, source_alpha, icc = load_canonical_png(self.store.canonical_path(project_id))
        rgb, _ = inference_srgb_copy(canonical_rgb, icc)
        before = self.store.read_alpha(project_id)
        locks = self.store.read_lock(project_id)
        after, bounds, operation = apply_magic_wand(
            rgb,
            before,
            source_alpha,
            locks,
            int(params["x"]),
            int(params["y"]),
            float(params.get("tolerance", 30.0)),
            float(params.get("softness", 18.0)),
            bool(params.get("contiguous", True)),
            params.get("mode", "remove"),
            params.get("wand_algorithm", "SMART"),
        )
        self.store.commit_alpha_edit(project_id, before, after, bounds, operation)
        self._refresh_preview(project_id, self.store.read_working_rgb(project_id), after)
        return self._project_payload(project_id)

    def preview_magic_wand(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = params["project_id"]
        canonical_rgb, source_alpha, icc = load_canonical_png(self.store.canonical_path(project_id))
        inference_rgb, _ = inference_srgb_copy(canonical_rgb, icc)
        selection = magic_wand_selection(
            inference_rgb,
            int(params["x"]),
            int(params["y"]),
            float(params.get("tolerance", 30.0)),
            float(params.get("softness", 18.0)),
            bool(params.get("contiguous", True)),
            algorithm=params.get("wand_algorithm", "SMART"),
        )
        token = uuid.uuid4().hex
        staging = self.store.path(project_id) / "alpha" / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        selection_path = staging / f"wand-{token}.npy"
        np.save(selection_path, selection.astype(np.float32), allow_pickle=False)
        before = self.store.read_alpha(project_id)
        locks = self.store.read_lock(project_id)
        preview_alpha, bounds, _ = apply_wand_coverage(
            before,
            source_alpha,
            locks,
            selection,
            params.get("mode", "remove"),
        )
        preview_path = self.store.preview_path(project_id, f"wand-preview-{token}")
        save_preview(self.store.read_working_rgb(project_id), preview_alpha, preview_path)
        return {
            "selection_id": token,
            "preview_path": str(preview_path.resolve()),
            "selected_pixel_count": int(np.count_nonzero(selection > 0.001)),
            "bounds": list(bounds),
            "mode": params.get("mode", "remove"),
            "wand_algorithm": str(params.get("wand_algorithm", "SMART")).upper(),
        }

    def commit_magic_wand(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = params["project_id"]
        token = str(params["selection_id"])
        if len(token) != 32 or any(char not in "0123456789abcdef" for char in token.lower()):
            raise ValueError("selection_id không hợp lệ")
        selection_path = self.store.path(project_id) / "alpha" / "staging" / f"wand-{token}.npy"
        if not selection_path.is_file():
            raise FileNotFoundError("Wand preview đã hết hạn hoặc không tồn tại")
        selection = np.load(selection_path, allow_pickle=False)
        canonical_rgb, source_alpha, _ = load_canonical_png(self.store.canonical_path(project_id))
        before = self.store.read_alpha(project_id)
        locks = self.store.read_lock(project_id)
        after, bounds, operation = apply_wand_coverage(
            before,
            source_alpha,
            locks,
            selection,
            params.get("mode", "remove"),
            {
                "selection_id": token,
                "wand_algorithm": str(params.get("wand_algorithm", "SMART")).upper(),
                "selected_pixel_count": int(np.count_nonzero(selection > 0.001)),
            },
        )
        self.store.commit_alpha_edit(project_id, before, after, bounds, operation)
        selection_path.unlink(missing_ok=True)
        self.store.preview_path(project_id, f"wand-preview-{token}").unlink(missing_ok=True)
        self._refresh_preview(project_id, self.store.read_working_rgb(project_id), after)
        return self._project_payload(project_id)

    def cancel_magic_wand(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = params["project_id"]
        token = str(params["selection_id"])
        if len(token) != 32 or any(char not in "0123456789abcdef" for char in token.lower()):
            raise ValueError("selection_id không hợp lệ")
        self.store.path(project_id).joinpath("alpha", "staging", f"wand-{token}.npy").unlink(
            missing_ok=True
        )
        self.store.preview_path(project_id, f"wand-preview-{token}").unlink(missing_ok=True)
        return {"cancelled": True, "selection_id": token}

    def set_subject_selection(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = params["project_id"]
        selected_ids = {int(value) for value in params.get("selected_subject_ids", [])}
        base = self.store.read_alpha(project_id, "base")
        before = self.store.read_alpha(project_id)
        after = select_components(base, selected_ids)
        operation = {
            "tool": "subject_selection",
            "selected_subject_ids": sorted(selected_ids),
            "algorithm_version": "component-selection-v3",
        }
        self.store.commit_alpha_edit(
            project_id,
            before,
            after,
            (0, 0, after.shape[1], after.shape[0]),
            operation,
        )
        manifest = self.store.manifest(project_id)
        processing = manifest.get("processing") or {}
        processing["subject_policy"] = "SELECTED"
        processing["selected_subject_ids"] = sorted(selected_ids)
        for subject in processing.get("subjects", []):
            subject["selected"] = int(subject.get("id", -1)) in selected_ids
        manifest["processing"] = processing
        self.store.update_manifest(project_id, manifest)
        self._refresh_preview(project_id, self.store.read_working_rgb(project_id), after)
        return self._project_payload(project_id)

    def undo(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = params["project_id"]
        entry = self.store.undo(project_id)
        if entry:
            rgb = self.store.read_working_rgb(project_id)
            self._refresh_preview(project_id, rgb, self.store.read_alpha(project_id))
        payload = self._project_payload(project_id)
        payload["history_action"] = "undo" if entry else "none"
        return payload

    def redo(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = params["project_id"]
        entry = self.store.redo(project_id)
        if entry:
            rgb = self.store.read_working_rgb(project_id)
            self._refresh_preview(project_id, rgb, self.store.read_alpha(project_id))
        payload = self._project_payload(project_id)
        payload["history_action"] = "redo" if entry else "none"
        return payload

    def preflight(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = params["project_id"]
        manifest = self.store.manifest(project_id)
        alpha = self.store.read_alpha(project_id)
        source = manifest["source"]
        uses_generic_size = "print_width" in params or "print_height" in params
        print_width = (
            params.get("print_width") if uses_generic_size else params.get("print_width_inch")
        )
        print_height = (
            params.get("print_height") if uses_generic_size else params.get("print_height_inch")
        )
        print_unit = params.get("print_unit", "inch") if uses_generic_size else "inch"
        report = run_preflight(
            alpha,
            source["width"],
            source["height"],
            print_width,
            print_height,
            color_profile="sRGB",
            source_converted=bool(source.get("conversion_flags")),
            print_unit=print_unit,
        )
        report["project_id"] = project_id
        report["output_mode"] = params.get("output_mode", "POD_READY")
        report_path = self.store.path(project_id) / "reports" / "preflight-latest.json"
        atomic_write_json(report_path, report)
        report["report_path"] = str(report_path)
        return report

    def export(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = params["project_id"]
        output_mode = params["output_mode"]
        _, _, icc = load_canonical_png(self.store.canonical_path(project_id))
        rgb = self.store.read_working_rgb(project_id)
        alpha = self.store.read_alpha(project_id)
        manifest = self.store.manifest(project_id)
        processing_diagnostics = (manifest.get("processing") or {}).get("diagnostics", {})
        background_rgb = (
            processing_diagnostics.get("background_model")
            or processing_diagnostics.get("background_palette_rgb")
            or processing_diagnostics.get("background_rgb")
        )
        result = export_image(
            output_mode,
            params["destination"],
            rgb,
            alpha,
            icc,
            background_rgb=background_rgb,
            settings=params.get("settings"),
        )
        if output_mode == "MASTER_SOURCE_FAITHFUL" and not self.store.retouch_path(project_id).exists():
            with Image.open(result["path"]) as exported:
                exported_rgb = np.asarray(exported.convert("RGBA"), dtype=np.uint8)[:, :, :3]
            result["rgb_integrity"] = bool(np.array_equal(rgb, exported_rgb))
            if not result["rgb_integrity"]:
                Path(result["path"]).unlink(missing_ok=True)
                raise ValueError("RGB integrity test thất bại; master export đã bị hủy")
        manifest["export_settings"][output_mode] = params.get("settings", {})
        self.store.update_manifest(project_id, manifest)
        return result

    def list_models(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"models": list_model_manifests(self.models_dir)}

    def install_model_pack(self, params: dict[str, Any]) -> dict[str, Any]:
        manifest = install_model_pack(Path(params["path"]), self.models_dir)
        return {"model": manifest, "models": list_model_manifests(self.models_dir)}

    def download_model_pack(self, params: dict[str, Any]) -> dict[str, Any]:
        manifest = download_model_pack(str(params["model_id"]), self.models_dir)
        return {"model": manifest, "models": list_model_manifests(self.models_dir)}

    def remove_model_pack(self, params: dict[str, Any]) -> dict[str, Any]:
        removed = remove_model_pack(str(params["model_id"]), self.models_dir)
        return {"removed": removed, "models": list_model_manifests(self.models_dir)}

    def _refresh_preview(
        self, project_id: str, rgb: np.ndarray, alpha: np.ndarray
    ) -> tuple[int, int]:
        return save_preview(rgb, alpha, self.store.preview_path(project_id))

    def _project_payload(self, project_id: str, preview_name: str = "current") -> dict[str, Any]:
        manifest = self.store.manifest(project_id)
        preview_path = self.store.preview_path(project_id, preview_name)
        if not preview_path.exists():
            preview_path = self.store.preview_path(project_id, "original")
        source = manifest["source"]
        current_alpha = self.store.read_alpha(project_id)
        components = analyze_components(current_alpha)
        processing = manifest.get("processing") or None
        processing_warnings = [
            item.get("message", str(item)) if isinstance(item, dict) else str(item)
            for item in (processing or {}).get("warnings", [])
        ]
        return {
            "project_id": project_id,
            "project_path": str(self.store.path(project_id)),
            "schema_version": manifest["schema_version"],
            "source_path": manifest["source_reference"],
            "manifest": manifest,
            "preview_path": str(preview_path.resolve()),
            "revision": str(uuid.uuid4()),
            "width": source["width"],
            "height": source["height"],
            "canonical": {
                "raw_hash": source["source_file_sha256"],
                "decoded_pixel_hash": source["canonical_pixels_sha256"],
                "width": source["width"],
                "height": source["height"],
                "original_orientation": source["original_orientation"],
                "canonical_orientation": source["canonical_orientation"],
                "source_mode": source["source_color_mode"],
                "source_has_alpha": source["has_source_alpha"],
                "icc_profile_present": bool(source.get("icc_hash")),
                "conversion_flags": source.get("conversion_flags", []),
            },
            "components": {
                "count": len(components),
                "suspicious_count": sum(bool(item["needs_review"]) for item in components),
                "components": components[:100],
            },
            "history": {
                "can_undo": int(manifest.get("history_cursor", 0)) > 0,
                "can_redo": int(manifest.get("history_cursor", 0))
                < len(manifest.get("history", [])),
                "cursor": manifest.get("history_cursor", 0),
                "length": len(manifest.get("history", [])),
            },
            "process": processing,
            "warnings": processing_warnings,
            "retouch": {"watermark_removed": self.store.retouch_path(project_id).exists()},
        }
