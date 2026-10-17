"""Shared runtime setup and inference-only checkpoint format."""
import hashlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '8')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '4')
sys.path.insert(0, str(ROOT / 'src' / 'dreamer'))

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
import dreamer
import models
import tools
import wrappers

UPSTREAM_COMMIT = '56d4d444dfd0582b0e79dab80aebbea74c0ce40d'

def configure(seed, precision=32):
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError('A GPU is required for this reproduction run')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    prec.set_global_policy('mixed_float16' if precision == 16 else 'float32')
    return gpus

def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def module_digest(module):
    digest = hashlib.sha256()
    for var in module.variables:
        digest.update(var.numpy().tobytes())
    return digest.hexdigest()

def policy_modules(agent):
    return {'encode': agent._encode, 'dynamics': agent._dynamics, 'actor': agent._actor}

def export_policy(agent, directory):
    arrays, inventory = {}, {}
    for name, module in policy_modules(agent).items():
        inventory[name] = []
        for index, variable in enumerate(module.variables):
            key = f'{name}_{index:04d}'
            arrays[key] = variable.numpy()
            inventory[name].append({'key': key, 'name': variable.name,
                                    'shape': list(variable.shape)})
    target = pathlib.Path(directory) / 'policy_weights.npz'
    np.savez_compressed(target, **arrays)
    metadata = {'format': 'dreamer_v1_policy_v1', 'variables': inventory,
                'sha256': sha256(target), 'upstream_commit': UPSTREAM_COMMIT}
    target.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    return metadata

class InferencePolicy:
    """Only encoder, RSSM and actor: no optimizer, replay buffer or train call."""
    def __init__(self, config, action_size):
        self.config = config
        self.action_size = action_size
        acts = {'elu': tf.nn.elu, 'relu': tf.nn.relu}
        self._encode = models.ConvEncoder(config.cnn_depth, acts[config.cnn_act])
        self._dynamics = models.RSSM(config.stoch_size, config.deter_size,
                                     config.deter_size, acts[config.dense_act])
        self._actor = models.ActionDecoder(action_size, 4, config.num_units,
                                           config.action_dist,
                                           init_std=config.action_init_std,
                                           act=acts[config.dense_act])
        self.action(tf.zeros([1, 64, 64, 3], tf.uint8), None)

    @tf.function(jit_compile=False)
    def action(self, images, state):
        dtype = prec.global_policy().compute_dtype
        if state is None:
            latent = self._dynamics.initial(1)
            previous_action = tf.zeros([1, self.action_size], dtype)
        else:
            latent, previous_action = state
        image = tf.cast(images, dtype) / 255.0 - 0.5
        embed = self._encode({'image': image})
        latent, _ = self._dynamics.obs_step(latent, previous_action, embed)
        action = self._actor(self._dynamics.get_feat(latent)).mode()
        return action, (latent, action)

    def load(self, directory):
        target = pathlib.Path(directory) / 'policy_weights.npz'
        metadata = json.loads(target.with_suffix('.json').read_text())
        assert metadata['format'] == 'dreamer_v1_policy_v1'
        assert metadata['sha256'] == sha256(target), 'Checkpoint checksum mismatch'
        max_difference = 0.0
        with np.load(target, allow_pickle=False) as saved:
            expected_keys = []
            for name, module in policy_modules(self).items():
                inventory = metadata['variables'][name]
                assert len(inventory) == len(module.variables), name
                for item, variable in zip(inventory, module.variables):
                    value = saved[item['key']]
                    expected_keys.append(item['key'])
                    assert list(value.shape) == list(variable.shape) == item['shape']
                    assert np.all(np.isfinite(value))
                    variable.assign(value)
                    max_difference = max(max_difference,
                                         float(np.max(np.abs(variable.numpy() - value))))
            assert set(expected_keys) == set(saved.files)
        assert max_difference == 0.0
        return {'checkpoint_sha256': metadata['sha256'],
                'restore_max_abs_difference': max_difference}
