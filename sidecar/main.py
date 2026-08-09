#!/usr/bin/env python3
import sys

from cutout_sidecar.protocol import serve_stdio
from cutout_sidecar.worker import serve_worker_stdio


def configure_stdio_utf8() -> None:
    # Giao thức sidecar luôn dùng JSON UTF-8, không phụ thuộc code page của Windows.
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    try:
        configure_stdio_utf8()
        if "--worker" in sys.argv:
            serve_worker_stdio()
        else:
            serve_stdio()
    except KeyboardInterrupt:
        pass
