from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any


class WorkerSupervisor:
    """Own one sequential worker and restart once after a crash/broken pipe."""

    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None

    def _command(self) -> list[str]:
        if getattr(sys, "frozen", False):
            return [sys.executable, "--worker"]
        worker_script = Path(__file__).resolve().parent / "worker.py"
        # Launch the module through a tiny import expression so package-relative imports work
        # regardless of the coordinator's current working directory.
        sidecar_root = str(worker_script.parent.parent)
        return [
            sys.executable,
            "-u",
            "-c",
            (
                "import sys;"
                f"sys.path.insert(0,{sidecar_root!r});"
                "from cutout_sidecar.worker import serve_worker_stdio;"
                "serve_worker_stdio()"
            ),
        ]

    def _spawn(self) -> subprocess.Popen[str]:
        self.close()
        self.process = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        return self.process

    def _send_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        process = self.process
        if process is None or process.poll() is not None:
            process = self._spawn()
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Worker thiếu stdio")
        process.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("Worker kết thúc trước khi trả response")
        response = json.loads(line)
        if not response.get("ok"):
            error = response.get("error") or {}
            raise RuntimeError(f"{error.get('message', 'Worker stage thất bại')}\n{error.get('details', '')}")
        return response["result"]

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = {"id": str(uuid.uuid4()), "method": method, "params": params}
        try:
            return self._send_once(payload)
        except (BrokenPipeError, OSError, RuntimeError, json.JSONDecodeError) as first_error:
            self.close()
            try:
                return self._send_once(payload)
            except Exception as retry_error:
                raise RuntimeError(
                    f"Worker crash và retry cùng profile thất bại. Lần đầu: {first_error}. Retry: {retry_error}"
                ) from retry_error

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)
        if process.stdout:
            process.stdout.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
