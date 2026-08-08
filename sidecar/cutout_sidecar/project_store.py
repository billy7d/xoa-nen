from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .image_core import CanonicalImage, save_canonical_png, save_preview


SCHEMA_VERSION = "2.2.0"
TILE_SIZE = 512


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, data)


def _tile_name(tile_y: int, tile_x: int) -> str:
    return f"y{tile_y:05d}_x{tile_x:05d}.f32.zlib"


def _encode_tile(tile: np.ndarray) -> bytes:
    payload = np.ascontiguousarray(tile, dtype="<f4").tobytes(order="C")
    return zlib.compress(payload, level=6)


def _decode_tile(payload: bytes, shape: tuple[int, int]) -> np.ndarray:
    decoded = zlib.decompress(payload)
    array = np.frombuffer(decoded, dtype="<f4")
    expected = shape[0] * shape[1]
    if array.size != expected:
        raise ValueError(f"Tile corrupt: expected {expected} floats, got {array.size}")
    return np.ascontiguousarray(array.reshape(shape), dtype=np.float32)


class ProjectStore:
    def __init__(self, root: Path | None = None) -> None:
        default_root = Path(__file__).resolve().parents[2] / "projects"
        self.root = Path(os.environ.get("CUTOUT_PROJECTS_DIR", root or default_root)).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, project_id: str) -> Path:
        if not project_id or any(char not in "0123456789abcdef-" for char in project_id.lower()):
            raise ValueError("project_id không hợp lệ")
        project_path = (self.root / f"{project_id}.cutoutproj").resolve()
        if self.root not in project_path.parents:
            raise ValueError("Project path vượt scope")
        return project_path

    def create(self, canonical: CanonicalImage) -> dict[str, Any]:
        project_id = str(uuid.uuid4())
        image_id = str(uuid.uuid4())
        project = self.path(project_id)
        (project / "source").mkdir(parents=True)
        (project / "alpha" / "base").mkdir(parents=True)
        (project / "alpha" / "current").mkdir(parents=True)
        (project / "locks").mkdir(parents=True)
        (project / "journal" / "deltas").mkdir(parents=True)
        (project / "previews").mkdir(parents=True)
        (project / "reports").mkdir(parents=True)

        canonical_path = project / "source" / "canonical.png"
        save_canonical_png(canonical, canonical_path)
        save_preview(canonical.rgb, canonical.source_alpha, project / "previews" / "original.png")

        initial_alpha = canonical.source_alpha.astype(np.float32, copy=True)
        self.write_alpha(project_id, "base", initial_alpha)
        self.write_alpha(project_id, "current", initial_alpha)
        self.write_lock(project_id, np.zeros(initial_alpha.shape, dtype=np.uint8))

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "app_version": "0.1.0",
            "project_id": project_id,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "source": canonical.to_manifest(image_id),
            "source_reference": str(canonical.source_path),
            "coordinate_convention": {
                "pixel_center": "x+0.5,y+0.5",
                "rectangle": "half-open",
            },
            "alpha_store": {
                "dtype": "float32",
                "tile_size": TILE_SIZE,
                "codec": "zlib-v1",
            },
            "history": [],
            "history_cursor": 0,
            "journal_sequence": 0,
            "processing": None,
            "model_manifests": [],
            "export_settings": {},
        }
        atomic_write_json(project / "manifest.json", manifest)
        atomic_write_json(
            project / "source" / "reference.json",
            {
                "path": str(canonical.source_path),
                "source_file_sha256": canonical.source_file_sha256,
                "canonical_pixels_sha256": canonical.canonical_pixels_sha256,
            },
        )
        return manifest

    def manifest(self, project_id: str) -> dict[str, Any]:
        manifest_path = self.path(project_id) / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy project {project_id}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def update_manifest(self, project_id: str, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = utc_now()
        atomic_write_json(self.path(project_id) / "manifest.json", manifest)

    def canonical_path(self, project_id: str) -> Path:
        return self.path(project_id) / "source" / "canonical.png"

    def preview_path(self, project_id: str, name: str = "current") -> Path:
        return self.path(project_id) / "previews" / f"{name}.png"

    def _array_directory(self, project_id: str, name: str) -> Path:
        if name not in {"base", "current"}:
            raise ValueError("Alpha store name không hợp lệ")
        return self.path(project_id) / "alpha" / name

    def write_alpha(self, project_id: str, name: str, alpha: np.ndarray) -> None:
        directory = self._array_directory(project_id, name)
        directory.mkdir(parents=True, exist_ok=True)
        height, width = alpha.shape
        index = {
            "shape": [height, width],
            "tile_size": TILE_SIZE,
            "dtype": "float32",
            "codec": "zlib-v1",
        }
        for y0 in range(0, height, TILE_SIZE):
            for x0 in range(0, width, TILE_SIZE):
                tile = alpha[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE]
                atomic_write_bytes(
                    directory / _tile_name(y0 // TILE_SIZE, x0 // TILE_SIZE),
                    _encode_tile(tile),
                )
        atomic_write_json(directory / "index.json", index)

    def read_alpha(self, project_id: str, name: str = "current") -> np.ndarray:
        directory = self._array_directory(project_id, name)
        index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        height, width = map(int, index["shape"])
        alpha = np.empty((height, width), dtype=np.float32)
        for y0 in range(0, height, TILE_SIZE):
            for x0 in range(0, width, TILE_SIZE):
                shape = (min(TILE_SIZE, height - y0), min(TILE_SIZE, width - x0))
                tile_path = directory / _tile_name(y0 // TILE_SIZE, x0 // TILE_SIZE)
                alpha[y0 : y0 + shape[0], x0 : x0 + shape[1]] = _decode_tile(
                    tile_path.read_bytes(), shape
                )
        return alpha

    def write_lock(self, project_id: str, locks: np.ndarray) -> None:
        path = self.path(project_id) / "locks" / "locks.u8.zlib"
        metadata = {"shape": list(locks.shape), "dtype": "uint8", "codec": "zlib-v1"}
        atomic_write_bytes(path, zlib.compress(np.ascontiguousarray(locks, dtype=np.uint8).tobytes()))
        atomic_write_json(path.with_suffix(".json"), metadata)

    def read_lock(self, project_id: str) -> np.ndarray:
        path = self.path(project_id) / "locks" / "locks.u8.zlib"
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        shape = tuple(metadata["shape"])
        raw = zlib.decompress(path.read_bytes())
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()

    def commit_alpha_edit(
        self,
        project_id: str,
        before: np.ndarray,
        after: np.ndarray,
        bounds: tuple[int, int, int, int],
        operation: dict[str, Any],
    ) -> dict[str, Any]:
        manifest = self.manifest(project_id)
        sequence = int(manifest.get("journal_sequence", 0)) + 1
        history = list(manifest.get("history", []))
        cursor = int(manifest.get("history_cursor", len(history)))
        history = history[:cursor]

        x0, y0, x1, y1 = bounds
        height, width = after.shape
        x0 = max(0, min(width, x0))
        y0 = max(0, min(height, y0))
        x1 = max(x0, min(width, x1))
        y1 = max(y0, min(height, y1))
        delta_dir = self.path(project_id) / "journal" / "deltas" / f"{sequence:08d}"
        delta_tiles: list[dict[str, Any]] = []

        if x1 > x0 and y1 > y0:
            for tile_y in range(y0 // TILE_SIZE, (y1 - 1) // TILE_SIZE + 1):
                for tile_x in range(x0 // TILE_SIZE, (x1 - 1) // TILE_SIZE + 1):
                    tx0 = tile_x * TILE_SIZE
                    ty0 = tile_y * TILE_SIZE
                    tx1 = min(width, tx0 + TILE_SIZE)
                    ty1 = min(height, ty0 + TILE_SIZE)
                    before_tile = before[ty0:ty1, tx0:tx1]
                    after_tile = after[ty0:ty1, tx0:tx1]
                    name = _tile_name(tile_y, tile_x)
                    atomic_write_bytes(delta_dir / f"{name}.before", _encode_tile(before_tile))
                    atomic_write_bytes(delta_dir / f"{name}.after", _encode_tile(after_tile))
                    atomic_write_bytes(
                        self._array_directory(project_id, "current") / name,
                        _encode_tile(after_tile),
                    )
                    delta_tiles.append(
                        {
                            "tile_x": tile_x,
                            "tile_y": tile_y,
                            "shape": [ty1 - ty0, tx1 - tx0],
                            "name": name,
                        }
                    )

        entry = {
            "sequence": sequence,
            "operation": operation,
            "bounds": [x0, y0, x1, y1],
            "tiles": delta_tiles,
            "created_at": utc_now(),
        }
        history.append(entry)
        manifest["history"] = history
        manifest["history_cursor"] = len(history)
        manifest["journal_sequence"] = sequence
        self.update_manifest(project_id, manifest)
        journal_path = self.path(project_id) / "journal" / "edits.jsonl"
        with journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def _apply_history_entry(self, project_id: str, entry: dict[str, Any], side: str) -> None:
        delta_dir = self.path(project_id) / "journal" / "deltas" / f"{entry['sequence']:08d}"
        current_dir = self._array_directory(project_id, "current")
        for tile in entry["tiles"]:
            payload = (delta_dir / f"{tile['name']}.{side}").read_bytes()
            atomic_write_bytes(current_dir / tile["name"], payload)

    def undo(self, project_id: str) -> dict[str, Any] | None:
        manifest = self.manifest(project_id)
        history = manifest.get("history", [])
        cursor = int(manifest.get("history_cursor", len(history)))
        if cursor <= 0:
            return None
        entry = history[cursor - 1]
        self._apply_history_entry(project_id, entry, "before")
        manifest["history_cursor"] = cursor - 1
        self.update_manifest(project_id, manifest)
        return entry

    def redo(self, project_id: str) -> dict[str, Any] | None:
        manifest = self.manifest(project_id)
        history = manifest.get("history", [])
        cursor = int(manifest.get("history_cursor", len(history)))
        if cursor >= len(history):
            return None
        entry = history[cursor]
        self._apply_history_entry(project_id, entry, "after")
        manifest["history_cursor"] = cursor + 1
        self.update_manifest(project_id, manifest)
        return entry

    def reset_history(self, project_id: str) -> None:
        manifest = self.manifest(project_id)
        manifest["history"] = []
        manifest["history_cursor"] = 0
        manifest["journal_sequence"] = 0
        self.update_manifest(project_id, manifest)
        delta_dir = self.path(project_id) / "journal" / "deltas"
        if delta_dir.exists():
            shutil.rmtree(delta_dir)
        delta_dir.mkdir(parents=True)
        atomic_write_bytes(self.path(project_id) / "journal" / "edits.jsonl", b"")

