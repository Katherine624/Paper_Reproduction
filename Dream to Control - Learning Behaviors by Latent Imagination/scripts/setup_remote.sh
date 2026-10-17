#!/usr/bin/env bash
# Linux x86_64, Python 3.10; install the versions validated on RTX 6000D.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
python3.10 -c 'import ctypes; ctypes.CDLL("libEGL.so.1"); ctypes.CDLL("libGL.so.1")' || {
  echo 'Install system libraries first: apt-get update && apt-get install -y libegl1 libgl1 libgles2'
  exit 1
}
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip check
source scripts/env.sh
python scripts/runtime_probe.py
python scripts/validate_core.py
