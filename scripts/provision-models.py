from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


def _configure_stdout_utf8() -> None:
    """Giữ log tiếng Việt ổn định trên Windows cp1258 và terminal UTF-8 của macOS."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="strict")


_configure_stdout_utf8()


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_ROOT = REPOSITORY_ROOT / "sidecar"
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from cutout_sidecar.models import inspect_installed_manifest  # noqa: E402


@dataclass(frozen=True)
class ModelSource:
    model_id: str
    url: str


MODEL_SOURCES = (
    ModelSource(
        "studioludens-birefnet-lite-512",
        "https://huggingface.co/studioludens/birefnet-lite-512/resolve/"
        "4a3c40c36c94093cc1e724d9ea428b8fa4b57dc7/onnx/model.onnx?download=true",
    ),
    ModelSource(
        "xenova-vitmatte-small-composition-1k",
        "https://huggingface.co/Xenova/vitmatte-small-composition-1k/resolve/"
        "6bc1297f6140f055a227b6d2cfe8c093281f35d2/onnx/model_quantized.onnx?download=true",
    ),
    ModelSource(
        "local-opencv-lama-watermark-512",
        "https://huggingface.co/opencv/inpainting_lama/resolve/"
        "aee6d22f0a13e5e35af1c9a1c3afd62841fc6f3f/inpainting_lama_2025jan.onnx"
        "?download=true",
    ),
)


def _manifest_path(model_id: str) -> Path:
    return REPOSITORY_ROOT / "model-manifests" / model_id / "manifest.json"


def _manifest(model_id: str) -> dict[str, object]:
    path = _manifest_path(model_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("model_id") != model_id:
        raise ValueError(f"Manifest không khớp model_id: {path}")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError(f"Manifest phải có đúng một artifact: {path}")
    artifact = artifacts[0]
    if not isinstance(artifact, dict) or artifact.get("filename") != "model.onnx":
        raise ValueError(f"Manifest chỉ được provision model.onnx: {path}")
    return data


def _expected_artifact(manifest: dict[str, object]) -> tuple[int, str]:
    artifact = manifest["artifacts"][0]  # type: ignore[index]
    return int(artifact["size"]), str(artifact["sha256"]).lower()


def _is_ready(destination: Path, expected_revision: object) -> bool:
    try:
        inspected = inspect_installed_manifest(destination / "manifest.json")
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return bool(
        inspected.get("runtime_ready")
        and inspected.get("revision") == expected_revision
        and inspected.get("signature_valid")
        and inspected.get("checksum_valid")
    )


def _download_verified(source: ModelSource, destination: Path, expected_size: int, expected_hash: str) -> None:
    request = urllib.request.Request(
        source.url,
        headers={"User-Agent": "Local-POD-Cutout-Editor-model-provision/1.0"},
    )
    digest = hashlib.sha256()
    downloaded = 0
    next_report = 32 * 1024 * 1024
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                print(f"  đã tải {downloaded / (1024 * 1024):.1f} MiB", flush=True)
                next_report += 32 * 1024 * 1024
    actual_hash = digest.hexdigest()
    if downloaded != expected_size:
        raise ValueError(
            f"Sai kích thước {source.model_id}: nhận {downloaded}, cần {expected_size}"
        )
    if actual_hash != expected_hash:
        raise ValueError(
            f"Sai SHA-256 {source.model_id}: nhận {actual_hash}, cần {expected_hash}"
        )


def _install(source: ModelSource, models_dir: Path, verify_only: bool) -> str:
    manifest = _manifest(source.model_id)
    destination = models_dir / source.model_id
    if _is_ready(destination, manifest.get("revision")):
        return "đã sẵn sàng"
    if verify_only:
        raise RuntimeError(f"Model chưa sẵn sàng: {source.model_id}")

    expected_size, expected_hash = _expected_artifact(manifest)
    stage = Path(tempfile.mkdtemp(prefix=f".{source.model_id}-", dir=models_dir))
    backup = models_dir / f".{source.model_id}.previous"
    try:
        print(f"Đang tải {source.model_id}...", flush=True)
        _download_verified(source, stage / "model.onnx", expected_size, expected_hash)
        shutil.copy2(_manifest_path(source.model_id), stage / "manifest.json")
        inspected = inspect_installed_manifest(stage / "manifest.json")
        if not inspected.get("runtime_ready"):
            raise RuntimeError(
                f"Pack {source.model_id} bị từ chối sau khi tải: {inspected.get('status')}"
            )

        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(stage, destination)
        except Exception:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return "đã tải và xác minh"
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tải model ONNX đã pin mà không lưu weights trong Git"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPOSITORY_ROOT / "models",
        help="Thư mục model local đích",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=[item.model_id for item in MODEL_SOURCES],
        help="Chỉ provision model được chọn; có thể lặp lại",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Chỉ xác minh, không tải model còn thiếu",
    )
    args = parser.parse_args()

    models_dir = args.models_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.model or [])
    sources = [item for item in MODEL_SOURCES if not selected or item.model_id in selected]
    results: dict[str, str] = {}
    for source in sources:
        results[source.model_id] = _install(source, models_dir, args.verify_only)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
