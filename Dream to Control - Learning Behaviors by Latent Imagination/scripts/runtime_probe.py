"""Verify GPU graph gradients and a full pixel-control episode before training."""
import json
import os
import pathlib
import platform
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
import numpy as np
import tensorflow as tf

out = pathlib.Path(os.environ.get("PROBE_OUTPUT", "output/runtime_probe"))
out.mkdir(parents=True, exist_ok=True)
gpus = tf.config.list_physical_devices("GPU")
assert gpus, "TensorFlow did not discover a GPU"
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_soft_device_placement(False)
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(4)
print("GPU_DISCOVERED", gpus, flush=True)

with tf.device("/GPU:0"):
    weights = tf.Variable(tf.random.normal([4, 4, 3, 32]))
    image = tf.random.uniform([16, 64, 64, 3])

@tf.function(jit_compile=False)
def gpu_gradient(image):
    with tf.device("/GPU:0"):
        with tf.GradientTape() as tape:
            encoded = tf.nn.conv2d(image, weights, strides=2, padding="VALID")
            loss = tf.reduce_mean(tf.square(encoded))
        return loss, tape.gradient(loss, weights)

start = time.monotonic()
loss, grad = gpu_gradient(image)
assert np.isfinite(loss.numpy()) and np.all(np.isfinite(grad.numpy()))
assert "GPU:0" in grad.device, grad.device
assert float(tf.linalg.global_norm([grad])) > 0
print("GPU_GRAPH_GRADIENT_PASS", float(loss), grad.device, flush=True)
report = {"python": platform.python_version(), "tensorflow": tf.__version__,
          "build": tf.sysconfig.get_build_info(), "gpu": str(gpus),
          "gradient_device": grad.device, "gradient_norm": float(tf.linalg.global_norm([grad])),
          "gpu_probe_seconds": time.monotonic() - start}

from dm_control import suite
import mujoco
import imageio.v2 as imageio
env = suite.load("walker", "walk", task_kwargs={"random": 123})
env.reset()
frame = env.physics.render(64, 64, camera_id=0)
assert frame.shape == (64, 64, 3) and frame.dtype == np.uint8
assert frame.std() > 1, "Rendered image appears blank"
imageio.imwrite(out / "walker_random.png", frame)
random = np.random.RandomState(123)
spec = env.action_spec()
total_return, steps, start = 0.0, 0, time.monotonic()
while True:
    step = env.step(random.uniform(spec.minimum, spec.maximum))
    total_return += step.reward or 0.0
    steps += 1
    if step.last():
        break
assert steps == 1000, steps
report.update({"mujoco": mujoco.__version__, "render_shape": list(frame.shape),
               "random_episode_steps": steps, "random_episode_return": total_return,
               "episode_seconds": time.monotonic() - start, "status": "passed"})
(out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
env.close()
