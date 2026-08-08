#!/usr/bin/env python3
import sys

from cutout_sidecar.protocol import serve_stdio
from cutout_sidecar.worker import serve_worker_stdio


if __name__ == "__main__":
    try:
        if "--worker" in sys.argv:
            serve_worker_stdio()
        else:
            serve_stdio()
    except KeyboardInterrupt:
        pass
