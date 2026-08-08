from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from .image_core import load_canonical_png
from .processor import artwork_alpha


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
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def process_request(request: dict[str, Any]) -> dict[str, Any]:
    method = request.get("method")
    if method != "process_artwork":
        raise ValueError(f"Worker method không hỗ trợ: {method}")
    params = request.get("params") or {}
    canonical_path = Path(params["canonical_path"]).expanduser().resolve()
    output_path = Path(params["output_path"]).expanduser().resolve()
    if output_path.suffix.lower() != ".npy":
        raise ValueError("Worker output phải là file .npy trong project staging")
    rgb, source_alpha, _ = load_canonical_png(canonical_path)
    alpha, diagnostics = artwork_alpha(
        rgb,
        source_alpha,
        tolerance=float(params.get("tolerance", 30.0)),
        softness=float(params.get("softness", 18.0)),
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

