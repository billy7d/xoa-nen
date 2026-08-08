#!/bin/sh
set -eu

BUNDLED_PYTHON="/Users/lanphuong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
if [ -n "${CUTOUT_TEST_PYTHON:-}" ]; then
  PYTHON_BIN="$CUTOUT_TEST_PYTHON"
elif [ -x "$BUNDLED_PYTHON" ]; then
  PYTHON_BIN="$BUNDLED_PYTHON"
else
  PYTHON_BIN="python3"
fi

PYTHONPATH="sidecar${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m unittest discover -s tests -v

