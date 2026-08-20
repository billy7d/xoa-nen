from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..project_store import atomic_write_json
from .mask import apply_stroke_to_mask, bounds_from_mask


def _safe_session_id(value: str) -> str:
    token = str(value)
    if len(token) != 32 or any(char not in "0123456789abcdef" for char in token.lower()):
        raise ValueError("watermark session_id không hợp lệ")
    return token.lower()


def _session_root(project_path: Path) -> Path:
    path = project_path / "retouch" / "staging"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_dir(project_path: Path, session_id: str) -> Path:
    return _session_root(project_path) / f"watermark-{_safe_session_id(session_id)}"


def _mask_path(directory: Path) -> Path:
    return directory / "mask.npy"


def _metadata_path(directory: Path) -> Path:
    return directory / "session.json"


def _overlay_path(directory: Path) -> Path:
    return directory / "mask-overlay.png"


def _preview_path(directory: Path) -> Path:
    return directory / "restored-preview.png"


def invalidate_preview(directory: Path, metadata: dict[str, Any]) -> None:
    """Hủy kết quả phục hồi cũ ngay khi mask hoặc cấu hình đã thay đổi."""
    _preview_path(directory).unlink(missing_ok=True)
    metadata["status"] = "EDITING"
    metadata.pop("preview_diagnostics", None)


def render_overlay(mask: np.ndarray, destination: Path, source: str = "MIXED") -> None:
    values = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    alpha = np.rint(np.clip(values * 190.0, 0.0, 190.0)).astype(np.uint8)
    overlay = np.zeros((values.shape[0], values.shape[1], 4), dtype=np.uint8)
    uncertain = (values >= 0.12) & (values < 0.55)
    core = values >= 0.55
    manual_color = np.array([74, 188, 255], dtype=np.uint8)
    auto_color = np.array([255, 73, 91], dtype=np.uint8)
    uncertain_color = np.array([246, 202, 82], dtype=np.uint8)
    if str(source).upper() == "MANUAL":
        overlay[:, :, :3] = manual_color
    else:
        overlay[:, :, :3] = auto_color
    overlay[uncertain, :3] = uncertain_color
    overlay[core, 3] = alpha[core]
    overlay[uncertain, 3] = np.maximum(alpha[uncertain], 78)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, "RGBA").save(destination, format="PNG")


def _payload(directory: Path, metadata: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    bounds = bounds_from_mask(mask, 0.01)
    overlay = _overlay_path(directory)
    if not overlay.exists():
        render_overlay(mask, overlay, str(metadata.get("source", "MIXED")))
    preview = _preview_path(directory)
    return {
        "session_id": metadata["session_id"],
        "project_id": metadata["project_id"],
        "quality": metadata.get("quality", "BALANCED"),
        "feather": metadata.get("feather", 8),
        "expand": metadata.get("expand", "MEDIUM"),
        "mask_preview_path": str(overlay.resolve()),
        "mask_pixel_count": int(np.count_nonzero(mask > 0.01)),
        "strong_pixel_count": int(np.count_nonzero(mask >= 0.55)),
        "bounds": list(bounds),
        "source": metadata.get("source", "EMPTY"),
        "status": metadata.get("status", "EDITING"),
        "preview_path": str(preview.resolve()) if preview.is_file() else None,
        "preview_diagnostics": metadata.get("preview_diagnostics"),
        "revision": str(uuid.uuid4()),
        "diagnostics": metadata.get("diagnostics", {}),
    }


def begin_session(
    project_path: Path,
    project_id: str,
    shape: tuple[int, int],
    quality: str = "BALANCED",
    feather: float = 8.0,
    expand: str = "MEDIUM",
    base_revision: tuple[int, int] | None = None,
) -> dict[str, Any]:
    session_id = uuid.uuid4().hex
    directory = _session_dir(project_path, session_id)
    directory.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(shape, dtype=np.float32)
    metadata = {
        "session_id": session_id,
        "project_id": project_id,
        "quality": str(quality).upper(),
        "feather": float(feather),
        "expand": str(expand).upper(),
        "source": "EMPTY",
        "status": "EDITING",
        "base_revision": list(base_revision or (0, 0)),
        "diagnostics": {},
    }
    np.save(_mask_path(directory), mask, allow_pickle=False)
    atomic_write_json(_metadata_path(directory), metadata)
    render_overlay(mask, _overlay_path(directory), "EMPTY")
    return _payload(directory, metadata, mask)


def load_session(project_path: Path, session_id: str) -> tuple[Path, dict[str, Any], np.ndarray]:
    directory = _session_dir(project_path, session_id)
    metadata_file = _metadata_path(directory)
    mask_file = _mask_path(directory)
    if not metadata_file.is_file() or not mask_file.is_file():
        raise FileNotFoundError("Watermark session đã hết hạn hoặc không tồn tại")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    mask = np.load(mask_file, allow_pickle=False).astype(np.float32)
    return directory, metadata, mask


def save_session(
    directory: Path,
    metadata: dict[str, Any],
    mask: np.ndarray,
    source: str | None = None,
) -> dict[str, Any]:
    if source:
        metadata["source"] = source
    np.save(_mask_path(directory), np.clip(mask, 0.0, 1.0).astype(np.float32), allow_pickle=False)
    atomic_write_json(_metadata_path(directory), metadata)
    render_overlay(mask, _overlay_path(directory), str(metadata.get("source", "MIXED")))
    return _payload(directory, metadata, mask)


def update_session_mask(
    project_path: Path,
    session_id: str,
    points: list[dict[str, Any]],
    radius: float,
    hardness: float,
    feather: float,
    mode: str,
) -> dict[str, Any]:
    directory, metadata, mask = load_session(project_path, session_id)
    updated, bounds, changed = apply_stroke_to_mask(mask, points, radius, hardness, feather, mode)
    diagnostics = dict(metadata.get("diagnostics", {}))
    diagnostics["last_stroke"] = {
        "mode": str(mode).upper(),
        "points": len(points),
        "changed_pixels": changed,
        "bounds": list(bounds),
    }
    metadata["diagnostics"] = diagnostics
    metadata["feather"] = float(feather)
    metadata["source"] = "MANUAL" if metadata.get("source") in {"EMPTY", "MANUAL"} else "MIXED"
    invalidate_preview(directory, metadata)
    return save_session(directory, metadata, updated)


def replace_session_mask(
    project_path: Path,
    session_id: str,
    mask: np.ndarray,
    diagnostics: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    directory, metadata, _ = load_session(project_path, session_id)
    metadata["diagnostics"] = diagnostics
    invalidate_preview(directory, metadata)
    return save_session(directory, metadata, np.asarray(mask, dtype=np.float32), source=source)


def cancel_session(project_path: Path, session_id: str) -> dict[str, Any]:
    directory = _session_dir(project_path, session_id)
    shutil.rmtree(directory, ignore_errors=True)
    return {"cancelled": True, "session_id": _safe_session_id(session_id)}
