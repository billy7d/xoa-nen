from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROVISIONAL_MODELS = [
    {
        "model_id": "ZhengPeng7/BiRefNet_dynamic-matting",
        "role": "base_alpha_proposal",
        "status": "not_installed",
        "installed": False,
        "revision": "QUALIFICATION_REQUIRED",
        "weight_sha256": None,
        "weight_size": None,
        "code_revision": "QUALIFICATION_REQUIRED",
        "code_license": "MIT",
        "weight_license": "VERIFY_EXACT_REVISION",
        "commercial_pod_allowed": False,
        "redistribution_allowed": False,
        "preprocess_version": "birefnet-dynamic-candidate-v1",
        "qualified_backends": [],
        "runtime_remote_code_allowed": False,
    },
    {
        "model_id": "facebook/sam2.1-hiera-base-plus",
        "role": "conditional_topology",
        "status": "not_installed",
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
        "qualified_backends": [],
        "runtime_remote_code_allowed": False,
    },
    {
        "model_id": "hustvl/vitmatte-base-composition-1k",
        "role": "roi_matting",
        "status": "not_installed",
        "installed": False,
        "revision": "QUALIFICATION_REQUIRED",
        "weight_sha256": None,
        "weight_size": None,
        "code_revision": "transformers-pinned-at-qualification",
        "code_license": "Apache-2.0",
        "weight_license": "Apache-2.0",
        "commercial_pod_allowed": True,
        "redistribution_allowed": False,
        "preprocess_version": "vitmatte-candidate-v1",
        "qualified_backends": [],
        "runtime_remote_code_allowed": False,
    },
]


def list_model_manifests(model_directory: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    if model_directory.is_dir():
        for manifest_path in sorted(model_directory.glob("*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                weight_path = manifest_path.parent / manifest["weight_filename"]
                if weight_path.is_file():
                    digest = hashlib.sha256(weight_path.read_bytes()).hexdigest()
                    manifest["checksum_valid"] = digest == manifest.get("sha256")
                    manifest["status"] = "ready" if manifest["checksum_valid"] else "corrupt"
                    manifest["installed"] = bool(manifest["checksum_valid"])
                else:
                    manifest["checksum_valid"] = False
                    manifest["status"] = "missing_weights"
                    manifest["installed"] = False
                manifests.append(manifest)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                manifests.append(
                    {
                        "model_id": manifest_path.parent.name,
                        "status": "invalid_manifest",
                        "error": str(error),
                    }
                )
    installed_ids = {manifest.get("model_id") for manifest in manifests}
    manifests.extend(model for model in PROVISIONAL_MODELS if model["model_id"] not in installed_ids)
    return manifests
