#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/pi05-repro
source env-model.sh
cd openpi
uv sync --frozen --no-dev
uv pip check
.venv/bin/python - <<'PY'
import jax, numpy as np
from openpi.training import config
print('JAX',jax.__version__)
print('DEVICES',jax.devices())
a=jax.numpy.ones((512,512),dtype=jax.numpy.bfloat16)
b=(a@a).block_until_ready()
assert np.all(np.asarray(b,dtype=np.float32)==512)
assert any(d.platform=='gpu' for d in jax.devices())
print('GPU_MATMUL_PASS')
cfg=config.get_config('pi05_libero')
print('PI05_CONFIG',cfg.name,cfg.model.pi05,cfg.model.action_horizon)
PY
