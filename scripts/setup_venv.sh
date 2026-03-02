#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
MODE="${1:-runtime}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter '$PYTHON_BIN' was not found." >&2
  exit 1
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

if [[ "$MODE" == "dev" ]]; then
  python -m pip install -r requirements-dev.txt
fi

echo
echo "Environment ready in $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"
