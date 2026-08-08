#!/bin/sh
set -eu

if [ -n "${CUTOUT_BUILD_PYTHON:-}" ]; then
  PYTHON_BIN="$CUTOUT_BUILD_PYTHON"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi
"$PYTHON_BIN" -c "import PIL, numpy, PyInstaller" >/dev/null 2>&1 || {
  echo "Thiếu Pillow, NumPy hoặc PyInstaller. Cài bằng: $PYTHON_BIN -m pip install -r sidecar/requirements-build.txt" >&2
  exit 1
}

PYINSTALLER_CONFIG_DIR="sidecar/build/pyinstaller-config" "$PYTHON_BIN" -m PyInstaller \
  --noconfirm \
  --clean \
  --onefile \
  --name cutout-sidecar \
  --distpath sidecar/dist \
  --workpath sidecar/build \
  --specpath sidecar/build \
  sidecar/main.py

echo "Sidecar standalone đã tạo trong sidecar/dist/."
