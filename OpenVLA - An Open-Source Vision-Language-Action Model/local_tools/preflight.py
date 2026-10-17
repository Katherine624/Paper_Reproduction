import json,os,sys,time
from pathlib import Path
root=Path(os.environ["OPENVLA_ROOT"])
cfg=root/"config/libero";cfg.mkdir(parents=True,exist_ok=True)
lib=root/"LIBERO/libero/libero"
paths={"benchmark_root":str(lib),"bddl_files":str(lib/"bddl_files"),
    "init_states":str(lib/"init_files"),"datasets":str(root/"data"),"assets":str(lib/"assets")}
(cfg/"config.yaml").write_text("\n".join(k+": "+v for k,v in paths.items())+"\n")
import torch
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import transformers,flash_attn,robosuite,mujoco
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env,get_libero_image,get_libero_dummy_action
suite=benchmark.get_benchmark_dict()["libero_spatial"]()
task=suite.get_task(0)
env,description=get_libero_env(task,"openvla",resolution=256)
env.reset();obs=env.set_init_state(suite.get_task_init_states(0)[0])
for _ in range(10):obs,_,_,_=env.step(get_libero_dummy_action("openvla"))
im=get_libero_image(obs,224)
assert im.shape==(224,224,3) and im.dtype==np.uint8
from PIL import Image
Image.fromarray(im).save(root/"results/simulator_preflight.png")
x=torch.randn(32,32,device="cuda")
assert torch.isfinite(x@x.T).all().item()
info={"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,
    "gpu":torch.cuda.get_device_name(0),"transformers":transformers.__version__,
    "tensorflow":tf.__version__,"flash_attn":flash_attn.__version__,
    "robosuite":robosuite.__version__,"mujoco":mujoco.__version__,
    "task_count":suite.n_tasks,"first_task":description,"image_shape":list(im.shape),"status":"passed"}
(root/"results/preflight.json").write_text(json.dumps(info,indent=2))
env.close();print(json.dumps(info),flush=True)
