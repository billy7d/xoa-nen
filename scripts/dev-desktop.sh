#!/bin/sh
set -eu

BUNDLED_PYTHON="/Users/lanphuong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
if [ -n "${CUTOUT_PYTHON:-}" ]; then
  PYTHON_BIN="$CUTOUT_PYTHON"
elif [ -x "$BUNDLED_PYTHON" ]; then
  PYTHON_BIN="$BUNDLED_PYTHON"
else
  PYTHON_BIN="python3"
fi

CUTOUT_PYTHON="$PYTHON_BIN" npm run tauri dev

