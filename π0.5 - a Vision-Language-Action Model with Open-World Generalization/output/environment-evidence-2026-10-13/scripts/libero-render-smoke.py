from pathlib import Path
import os
import numpy as np
import imageio.v2 as imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = benchmark.get_benchmark_dict()['libero_spatial']()
task = suite.get_task(0)
bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
print('TASK', task.language, flush=True)
env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
try:
    env.seed(7)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(0)[0])
    for _ in range(10):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    for key in ('agentview_image', 'robot0_eye_in_hand_image'):
        image = obs[key]
        assert image.shape == (256, 256, 3), (key, image.shape)
        assert np.isfinite(image).all() and image.std() > 1, key
        dest = Path(os.environ['PI05_ROOT']) / 'results' / (key + '.png')
        imageio.imwrite(dest, np.ascontiguousarray(image[::-1, ::-1]))
        print('RENDER', key, image.shape, 'SAVED', str(dest), flush=True)
    print('PASS: simulation reset, 10 physics steps and two camera renders; no policy evaluation.', flush=True)
finally:
    env.close()
