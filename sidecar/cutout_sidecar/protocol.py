from __future__ import annotations

import json
import sys
import traceback
from typing import Any, TextIO

from .coordinator import Coordinator


def _configure_utf8_system_stream(stream: TextIO, system_stream: TextIO) -> None:
    """Giữ giao thức JSON Unicode ổn định khi app chạy trong Windows console."""
    if stream is not system_stream:
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="strict")
        except (OSError, ValueError):
            pass


def handle_request(coordinator: Coordinator, request: dict[str, Any]) -> dict[str, Any]:
    request_id = request.get("id")
    try:
        method = request.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError("Request thiếu method")
        params = request.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("params phải là object")
        result = coordinator.dispatch(method, params)
        return {"id": request_id, "ok": True, "result": result}
    except Exception as error:  # boundary: errors must cross stdio as structured JSON
        return {
            "id": request_id,
            "ok": False,
            "error": {
                "code": type(error).__name__.upper(),
                "message": str(error),
                "details": traceback.format_exc(limit=8),
            },
        }


def serve_stdio(input_stream: TextIO | None = None, output_stream: TextIO | None = None) -> None:
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    _configure_utf8_system_stream(input_stream, sys.stdin)
    _configure_utf8_system_stream(output_stream, sys.stdout)
    coordinator = Coordinator()
    for line in input_stream:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            request = json.loads(stripped)
            if not isinstance(request, dict):
                raise ValueError("Request root phải là object")
            response = handle_request(coordinator, request)
        except Exception as error:
            response = {
                "id": None,
                "ok": False,
                "error": {"code": "INVALID_JSON", "message": str(error)},
            }
        output_stream.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
        output_stream.flush()
