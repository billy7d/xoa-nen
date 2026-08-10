from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
except ImportError:  # pragma: no cover - release dependency, explicit status in source mode.
    InvalidSignature = ValueError  # type: ignore[assignment,misc]
    Ed25519PublicKey = None  # type: ignore[assignment]


MODEL_PACK_SCHEMA = "1.0.0"
# Khóa ký chỉ chứa public key; private key không nằm trong repo hoặc trong app.
# Khóa local này dùng để xác minh hai artifact ONNX đã qualification sơ bộ trên máy.
TRUSTED_SIGNING_KEYS: dict[str, str] = {
    "local-qualified-2026-08": "+E8YMcnYkQGw/jeE8dmmY7C0CUfQZEH+Y+9Rjye8Rf4=",
}

PROVISIONAL_MODELS = [
    {
        "model_id": "ZhengPeng7/BiRefNet_lite-matting",
        "role": "base_alpha_proposal",
        "adapter": "birefnet-v1",
        "status": "qualification_required",
        "installed": False,
        "revision": "PIN_EXACT_REVISION_DURING_QUALIFICATION",
        "weight_sha256": None,
        "weight_size": 177_600_000,
        "code_revision": "VENDORED_ONNX_ADAPTER_REQUIRED",
        "code_license": "MIT",
        "weight_license": "MIT_MODEL_CARD_VERIFY_EXACT_REVISION",
        "commercial_pod_allowed": False,
        "redistribution_allowed": False,
        "preprocess_version": "birefnet-lite-candidate-v1",
        "priority": 100,
        "input_size": [1024, 1024],
        "input_layout": "NCHW",
        "normalization": "imagenet",
        "output_layout": "NCHW",
        "output_activation": "auto",
        "output_semantics": "foreground",
        "qualified_backends": [],
        "runtime_remote_code_allowed": False,
        "download_url": None,
    },
    {
        "model_id": "hustvl/vitmatte-small-composition-1k",
        "role": "roi_matting",
        "adapter": "vitmatte-v1",
        "status": "qualification_required",
        "installed": False,
        "revision": "PIN_EXACT_REVISION_DURING_QUALIFICATION",
        "weight_sha256": None,
        "weight_size": 103_000_000,
        "code_revision": "VENDORED_ONNX_ADAPTER_REQUIRED",
        "code_license": "Apache-2.0",
        "weight_license": "Apache-2.0",
        "commercial_pod_allowed": True,
        "redistribution_allowed": False,
        "preprocess_version": "vitmatte-small-candidate-v1",
        "priority": 100,
        "input_size": [512, 512],
        "input_layout": "NCHW",
        "normalization": "imagenet",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "trimap_mode": "concat",
        "output_layout": "NCHW",
        "output_activation": "auto",
        "output_semantics": "foreground",
        "qualified_backends": [],
        "runtime_remote_code_allowed": False,
        "download_url": None,
    },
    {
        "model_id": "facebook/sam2.1-hiera-base-plus",
        "role": "conditional_topology",
        "adapter": "sam2-conditional-v1",
        "status": "challenger_not_release_blocking",
        "installed": False,
        "revision": "QUALIFICATION_REQUIRED",
        "weight_sha256": None,
        "weight_size": None,
        "code_revision": "QUALIFICATION_REQUIRED",
        "code_license": "Apache-2.0",
        "weight_license": "Apache-2.0",
        "commercial_pod_allowed": True,
        "redistribution_allowed": False,
        "preprocess_version": "sam2.1-candidate-v1",
        "priority": 50,
        "input_size": [1024, 1024],
        "input_layout": "NCHW",
        "normalization": "imagenet",
        "prompt_mode": "mask",
        "output_layout": "NCHW",
        "output_activation": "auto",
        "output_semantics": "foreground",
        "qualified_backends": [],
        "runtime_remote_code_allowed": False,
        "download_url": None,
    },
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _signature_payload(manifest: dict[str, Any]) -> bytes:
    unsigned = {key: value for key, value in manifest.items() if key != "signature_ed25519"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _signature_valid(manifest: dict[str, Any]) -> bool:
    if Ed25519PublicKey is None:
        return False
    key_id = str(manifest.get("signature_key_id", ""))
    encoded_key = TRUSTED_SIGNING_KEYS.get(key_id)
    encoded_signature = manifest.get("signature_ed25519")
    if not encoded_key or not isinstance(encoded_signature, str):
        return False
    try:
        public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(encoded_key))
        public_key.verify(base64.b64decode(encoded_signature), _signature_payload(manifest))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def _artifact_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        return [dict(item) for item in artifacts if isinstance(item, dict)]
    if manifest.get("weight_filename"):
        return [
            {
                "filename": manifest["weight_filename"],
                "sha256": manifest.get("sha256") or manifest.get("weight_sha256"),
                "size": manifest.get("weight_size"),
                "role": manifest.get("role", "base_alpha_proposal"),
                "adapter": manifest.get("adapter"),
            }
        ]
    return []


def inspect_installed_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["signature_valid"] = _signature_valid(manifest)
    manifest["install_path"] = str(manifest_path.parent.resolve())
    artifacts = _artifact_entries(manifest)
    artifact_status: list[dict[str, Any]] = []
    all_valid = bool(artifacts)
    for artifact in artifacts:
        filename = str(artifact.get("filename", ""))
        path = (manifest_path.parent / filename).resolve()
        in_pack = manifest_path.parent.resolve() in path.parents
        exists = in_pack and path.is_file()
        digest = _sha256_file(path) if exists else None
        expected_size = artifact.get("size")
        size_valid = exists and (
            expected_size in (None, 0) or path.stat().st_size == int(expected_size)
        )
        checksum_valid = exists and digest == artifact.get("sha256")
        valid = bool(size_valid and checksum_valid)
        all_valid = all_valid and valid
        artifact_status.append(
            {
                "filename": filename,
                "exists": exists,
                "checksum_valid": checksum_valid,
                "size_valid": size_valid,
            }
        )
    policy_valid = bool(
        manifest.get("commercial_pod_allowed")
        and manifest.get("redistribution_allowed")
        and not manifest.get("runtime_remote_code_allowed", False)
        and manifest.get("qualified_backends")
    )
    ready = bool(all_valid and manifest["signature_valid"] and policy_valid)
    manifest["artifacts_status"] = artifact_status
    manifest["checksum_valid"] = all_valid
    manifest["policy_valid"] = policy_valid
    manifest["installed"] = ready
    if ready:
        manifest["status"] = "ready"
    elif not manifest["signature_valid"]:
        manifest["status"] = "untrusted_signature"
    elif not all_valid:
        manifest["status"] = "corrupt_or_missing_artifact"
    else:
        manifest["status"] = "policy_or_backend_not_qualified"
    return manifest


def list_model_manifests(model_directory: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    if model_directory.is_dir():
        for manifest_path in sorted(model_directory.glob("*/manifest.json")):
            try:
                manifests.append(inspect_installed_manifest(manifest_path))
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                manifests.append(
                    {
                        "model_id": manifest_path.parent.name,
                        "installed": False,
                        "status": "invalid_manifest",
                        "error": str(error),
                    }
                )
    installed_ids = {manifest.get("model_id") for manifest in manifests}
    manifests.extend(dict(model) for model in PROVISIONAL_MODELS if model["model_id"] not in installed_ids)
    return manifests


def _safe_zip_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    for item in archive.infolist():
        pure = PurePosixPath(item.filename)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise ValueError(f"Model-pack chứa path không an toàn: {item.filename}")
        if item.is_dir():
            continue
        members.append(item)
    return members


def install_model_pack(source: Path, model_directory: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".cutout-modelpack":
        raise ValueError("Cần file .cutout-modelpack hợp lệ")
    model_directory.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".modelpack-", dir=model_directory))
    try:
        with zipfile.ZipFile(source) as archive:
            members = _safe_zip_members(archive)
            archive.extractall(temporary_root, members=members)
        manifests = list(temporary_root.rglob("manifest.json"))
        if len(manifests) != 1:
            raise ValueError("Model-pack phải chứa đúng một manifest.json")
        manifest_path = manifests[0]
        pack_root = manifest_path.parent
        manifest = inspect_installed_manifest(manifest_path)
        if not manifest.get("installed"):
            raise ValueError(f"Model-pack bị từ chối: {manifest.get('status')}")
        model_id = str(manifest.get("model_id", "")).strip()
        if not model_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in model_id):
            raise ValueError("model_id không hợp lệ; chỉ dùng chữ, số, '.', '_' hoặc '-'")
        destination = model_directory / model_id
        backup = model_directory / f".{model_id}.previous"
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(pack_root, destination)
        except Exception:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return inspect_installed_manifest(destination / "manifest.json")
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def download_model_pack(model_id: str, model_directory: Path) -> dict[str, Any]:
    catalog = next((item for item in PROVISIONAL_MODELS if item["model_id"] == model_id), None)
    if not catalog or not catalog.get("download_url"):
        raise ValueError("Model chưa có artifact đã qualification để tải an toàn")
    url = str(catalog["download_url"])
    if not url.startswith("https://"):
        raise ValueError("Model catalog chỉ cho phép HTTPS")
    model_directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".cutout-modelpack", dir=model_directory)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        return install_model_pack(temporary, model_directory)
    finally:
        temporary.unlink(missing_ok=True)


def remove_model_pack(model_id: str, model_directory: Path) -> bool:
    if not model_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in model_id):
        raise ValueError("model_id không hợp lệ")
    destination = (model_directory / model_id).resolve()
    if model_directory.resolve() not in destination.parents:
        raise ValueError("Model path vượt scope")
    if not destination.exists():
        return False
    shutil.rmtree(destination)
    return True
