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
from .edits import apply_brush, apply_magic_wand
from .exports import export_image
from .image_core import decode_canonical, load_canonical_png, save_preview
from .models import list_model_manifests
from .preflight import run_preflight
from .processor import analyze_components
from .project_store import ProjectStore, atomic_write_json
from .worker_supervisor import WorkerSupervisor


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
            "apply_brush": self.brush,
            "apply_magic_wand": self.magic_wand,
            "undo": self.undo,
            "redo": self.redo,
            "preflight": self.preflight,
            "export": self.export,
            "list_models": self.list_models,
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
            "processing_engine": "classical-artwork-v1",
        }

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
        staging_path = self.store.path(project_id) / "alpha" / "staging" / "worker-alpha.npy"
        worker_result = self.worker.request(
            "process_artwork",
            {
                "canonical_path": str(self.store.canonical_path(project_id)),
                "output_path": str(staging_path),
                "tolerance": float(params.get("tolerance", 30.0)),
                "softness": float(params.get("softness", 18.0)),
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
        manifest = self.store.manifest(project_id)
        manifest["processing"] = {
            "content_mode": "ARTWORK",
            "quality_preset": params.get("quality_preset", "QUALITY"),
            "subject_policy": "PRESERVE_COMPONENTS",
            "diagnostics": diagnostics,
            "ai_models_used": [],
            "warnings": [
                {
                    "code": "AI_MODELS_NOT_INSTALLED",
                    "message": "Đang dùng Artwork Color/Edge engine local. Model AI sẽ được bật sau qualification/install.",
                }
            ],
        }
        self.store.update_manifest(project_id, manifest)
        self._refresh_preview(project_id, rgb, alpha)
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
        self._refresh_preview(project_id, rgb, after)
        return self._project_payload(project_id)

    def magic_wand(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = params["project_id"]
        rgb, source_alpha, _ = load_canonical_png(self.store.canonical_path(project_id))
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
        )
        self.store.commit_alpha_edit(project_id, before, after, bounds, operation)
        self._refresh_preview(project_id, rgb, after)
        return self._project_payload(project_id)

    def undo(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = params["project_id"]
        entry = self.store.undo(project_id)
        if entry:
            rgb, _, _ = load_canonical_png(self.store.canonical_path(project_id))
            self._refresh_preview(project_id, rgb, self.store.read_alpha(project_id))
        payload = self._project_payload(project_id)
        payload["history_action"] = "undo" if entry else "none"
        return payload

    def redo(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = params["project_id"]
        entry = self.store.redo(project_id)
        if entry:
            rgb, _, _ = load_canonical_png(self.store.canonical_path(project_id))
            self._refresh_preview(project_id, rgb, self.store.read_alpha(project_id))
        payload = self._project_payload(project_id)
        payload["history_action"] = "redo" if entry else "none"
        return payload

    def preflight(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = params["project_id"]
        manifest = self.store.manifest(project_id)
        alpha = self.store.read_alpha(project_id)
        source = manifest["source"]
        report = run_preflight(
            alpha,
            source["width"],
            source["height"],
            params.get("print_width_inch"),
            params.get("print_height_inch"),
            color_profile="sRGB",
            source_converted=bool(source.get("conversion_flags")),
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
        rgb, _, icc = load_canonical_png(self.store.canonical_path(project_id))
        alpha = self.store.read_alpha(project_id)
        manifest = self.store.manifest(project_id)
        background_rgb = (
            (manifest.get("processing") or {}).get("diagnostics", {}).get("background_rgb")
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
        if output_mode == "MASTER_SOURCE_FAITHFUL":
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
        }
