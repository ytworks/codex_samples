#!/usr/bin/env bash
set -euo pipefail

echo "[setup] Detecting environment manager..."

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

has_cmd() { command -v "$1" >/dev/null 2>&1; }

if has_cmd mamba; then
  echo "[setup] Using mamba with environment.yml"
  mamba env update -f "$ROOT_DIR/environment.yml" || mamba env create -f "$ROOT_DIR/environment.yml"
  echo "[setup] Done. Activate with: conda activate phscreen"
elif has_cmd conda; then
  echo "[setup] Using conda with environment.yml"
  conda env update -f "$ROOT_DIR/environment.yml" || conda env create -f "$ROOT_DIR/environment.yml"
  echo "[setup] Done. Activate with: conda activate phscreen"
else
  echo "[setup] No conda/mamba found. Falling back to venv + pip."
  PY=python3
  if ! has_cmd python3; then PY=python; fi
  "$PY" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install -U pip
  echo "[setup] Installing runtime dependencies"
  pip install -r "$ROOT_DIR/requirements.txt"
  echo "[setup] Installing dev dependencies"
  pip install -r "$ROOT_DIR/requirements-dev.txt"
  echo "[setup] Editable install of the package"
  pip install -e "$ROOT_DIR"[dev]
  echo "[setup] Done. Activate with: source .venv/bin/activate"
fi

