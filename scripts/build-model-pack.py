from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Đóng gói ONNX thành cutout-modelpack đã ký")
    parser.add_argument("--source", type=Path, required=True, help="Đường dẫn model.onnx nguồn")
    parser.add_argument("--destination", type=Path, required=True, help="File .cutout-modelpack đầu ra")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--preprocess-version", required=True)
    parser.add_argument("--code-license", default="MIT")
    parser.add_argument("--weight-license", required=True)
    parser.add_argument("--input-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), required=True)
    parser.add_argument("--input-name", required=True)
    parser.add_argument("--image-input-name", default=None)
    parser.add_argument("--mask-input-name", default=None)
    parser.add_argument("--mask-input-layout", choices=("NCHW", "NHWC"), default=None)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--mean", type=float, nargs=3, metavar=("R", "G", "B"), required=True)
    parser.add_argument("--std", type=float, nargs=3, metavar=("R", "G", "B"), required=True)
    parser.add_argument("--priority", type=int, default=150)
    parser.add_argument("--output-activation", default="auto")
    parser.add_argument("--normalization", default="imagenet")
    parser.add_argument("--output-range", default="zero_one")
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--trimap-mode", default=None)
    parser.add_argument("--qualified-backend", action="append", default=["CPUExecutionProvider"])
    return parser


def main() -> int:
    args = _parser().parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".onnx":
        raise ValueError("Source phải là một file .onnx tồn tại")
    if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in args.model_id):
        raise ValueError("model_id chỉ được chứa chữ, số, '.', '_' hoặc '-'")
    private_value = os.environ.get("CUTOUT_MODEL_PACK_PRIVATE_KEY", "")
    key_id = os.environ.get("CUTOUT_MODEL_PACK_KEY_ID", "local-qualified-2026-08")
    if not private_value:
        raise ValueError("Thiếu biến môi trường CUTOUT_MODEL_PACK_PRIVATE_KEY")
    private_key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_value))
    public_key = base64.b64encode(
        private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    ).decode("ascii")
    artifact_size = source.stat().st_size
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "model_id": args.model_id,
        "revision": args.revision,
        "role": args.role,
        "adapter": args.adapter,
        "status": "qualification_required",
        "commercial_pod_allowed": True,
        "redistribution_allowed": True,
        "runtime_remote_code_allowed": False,
        "code_revision": "builtin-onnx-adapter-v1",
        "code_license": args.code_license,
        "weight_license": args.weight_license,
        "preprocess_version": args.preprocess_version,
        "priority": args.priority,
        "input_size": list(args.input_size),
        "input_layout": "NCHW",
        "normalization": args.normalization,
        "mean": list(args.mean),
        "std": list(args.std),
        "input_name": args.input_name,
        "output_name": args.output_name,
        "output_layout": "NCHW",
        "output_activation": args.output_activation,
        "output_range": args.output_range,
        "output_semantics": "foreground",
        "qualified_backends": list(dict.fromkeys(args.qualified_backend)),
        "signature_key_id": key_id,
        "artifacts": [
            {
                "filename": "model.onnx",
                "sha256": _sha256_file(source),
                "size": artifact_size,
                "role": args.role,
                "adapter": args.adapter,
            }
        ],
    }
    if args.trimap_mode:
        manifest["trimap_mode"] = args.trimap_mode
    if args.image_input_name:
        # Lưu tên input ảnh riêng để runtime hỗ trợ ONNX nhiều input một cách tường minh.
        manifest["image_input_name"] = args.image_input_name
    if args.mask_input_name:
        manifest["mask_input_name"] = args.mask_input_name
    if args.mask_input_layout:
        manifest["mask_input_layout"] = args.mask_input_layout
    if args.source_url:
        manifest["source_url"] = args.source_url
    manifest["signature_ed25519"] = base64.b64encode(
        private_key.sign(_signature_payload(manifest))
    ).decode("ascii")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("pack/manifest.json", manifest_json)
        archive.write(source, "pack/model.onnx")
    print(f"model_pack={destination}")
    print(f"model_id={args.model_id}")
    print(f"artifact_sha256={manifest['artifacts'][0]['sha256']}")
    print(f"signature_key_id={key_id}")
    print(f"public_key={public_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
