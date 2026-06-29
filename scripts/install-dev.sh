#!/usr/bin/env bash
set -euo pipefail

# cd to repo root
SCRIPT_PATH="${BASH_SOURCE:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
cd "$SCRIPT_DIR/.."

# Make uv install into the current interpreter (conda/venv)
export UV_PYTHON="$(python -c 'import sys; print(sys.executable)')"

OS="$(uname -s || echo unknown)"

case "$OS" in
  Linux*)
    if [ "${CI:-false}" = "true" ]; then
      # Linux CI: use CPU-only PyTorch wheels from the dedicated index.
      # This avoids downloading the ~2GB CUDA wheels — CI only needs CPU.
      uv pip install \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu
    else
      # Local Linux dev: install default PyPI wheels which include CUDA support
      echo "Installing CUDA-enabled PyTorch for local development..."
      uv pip install torch torchvision
    fi
    ;;

  Darwin*)
    # macOS: PyPI wheels are CPU-only by default, no special index needed
    ;;

  MINGW*|MSYS*|CYGWIN*)
    # Windows (Git Bash/MSYS): PyPI wheels are CPU-only by default
    ;;

  *)
    echo "Unsupported OS: $OS" >&2
    exit 1
    ;;
esac

pip install -e ".[dev]"
