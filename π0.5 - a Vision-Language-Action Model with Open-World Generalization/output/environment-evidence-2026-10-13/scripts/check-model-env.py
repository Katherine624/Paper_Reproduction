import jax
import numpy as np
print('JAX', jax.__version__, flush=True)
print('DEVICES', jax.devices(), flush=True)
a = jax.numpy.ones((512, 512), dtype=jax.numpy.bfloat16)
b = (a @ a).block_until_ready()
assert np.all(np.asarray(b, dtype=np.float32) == 512)
assert any(d.platform == 'gpu' for d in jax.devices())
print('GPU_MATMUL_PASS', flush=True)
from openpi.training import config
cfg = config.get_config('pi05_libero')
print('PI05_CONFIG', cfg.name, cfg.model.pi05, cfg.model.action_horizon, flush=True)
