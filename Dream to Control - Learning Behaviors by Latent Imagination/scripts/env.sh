#!/usr/bin/env bash
# Source this file from the cloud checkout before invoking the scripts.
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$TASK_ROOT/.venv/bin/activate"
export TF_USE_LEGACY_KERAS=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8
export TF_NUM_INTEROP_THREADS=4
export PYTHONUNBUFFERED=1
export CUDA_CACHE_PATH="$TASK_ROOT/.cuda-cache"
export CUDA_CACHE_MAXSIZE=2147483648
mkdir -p "$CUDA_CACHE_PATH"
TASK_SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
export PATH="$TASK_SITE_PACKAGES/nvidia/cuda_nvcc/bin:$PATH"
TASK_CUDA_LIBS="$(find "$TASK_SITE_PACKAGES/nvidia" -type d -name lib -printf '%p:' 2>/dev/null)"
export LD_LIBRARY_PATH="$TASK_ROOT/runtime-libs/usr/lib/x86_64-linux-gnu:$TASK_CUDA_LIBS${LD_LIBRARY_PATH:-}"
