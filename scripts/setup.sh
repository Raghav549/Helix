#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [ "${HELIX_TORCH_SOURCE:-cpu}" = cpu ]; then
  # This is the official PyTorch index. It is not reachable in every sandbox.
  .venv/bin/python -m pip install --no-deps 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
  # The CPU wheel above is deliberately installed without dependencies so it
  # cannot be replaced by a PyPI CUDA wheel. Resolve its own metadata against
  # PyPI afterwards; the installed +cpu version already satisfies ==2.5.1.
  .venv/bin/python -m pip install 'torch==2.5.1'
elif [ "$HELIX_TORCH_SOURCE" = pypi ]; then
  echo 'Installing PyPI torch; Linux wheels may include multi-GB CUDA dependencies.' >&2
  .venv/bin/python -m pip install 'torch==2.5.1'
else
  echo 'HELIX_TORCH_SOURCE must be cpu or pypi' >&2; exit 1
fi
.venv/bin/python -m pip install -r requirements.txt 'pypdf==6.21.3'
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python -m helix doctor
