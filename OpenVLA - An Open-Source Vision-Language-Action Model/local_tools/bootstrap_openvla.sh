set -euo pipefail
PROJECT=/root/gpufree-data/openvla-repro
UV=/root/gpufree-data/pi05-repro/tools/bin/uv
export UV_CACHE_DIR="$PROJECT/cache/uv"
export UV_HTTP_TIMEOUT=120
export UV_CONCURRENT_DOWNLOADS=32
export UV_LINK_MODE=copy
mkdir -p "$PROJECT"/{logs,scripts,models,data,results,cache,envs}
cd "$PROJECT"
test -x envs/openvla/bin/python || "$UV" venv envs/openvla --python /opt/conda/bin/python
export VIRTUAL_ENV="$PROJECT/envs/openvla"
export PATH="$VIRTUAL_ENV/bin:$PATH"
"$UV" pip install --find-links "$PROJECT/wheels" --index-url https://mirrors.aliyun.com/pypi/simple "torch==2.2.0" "torchvision==0.17.0" "torchaudio==2.2.0" "numpy==1.26.4" "huggingface_hub==0.25.2" "packaging" "ninja" "setuptools" "wheel"
test -d openvla/.git || git clone --depth 1 https://github.com/openvla/openvla.git openvla
test -d LIBERO/.git || git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
test -d dlimp/.git || git clone --depth 1 https://github.com/moojink/dlimp_openvla.git dlimp
git -C openvla rev-parse HEAD > results/openvla_commit.txt
git -C LIBERO rev-parse HEAD > results/libero_commit.txt
git -C dlimp rev-parse HEAD > results/dlimp_commit.txt
"$UV" pip install --find-links "$PROJECT/wheels" --index-url https://mirrors.aliyun.com/pypi/simple "accelerate==0.30.1" "draccus==0.8.0" einops json-numpy jsonlines "matplotlib<3.10" "peft==0.11.1" "protobuf==3.20.3" rich "sentencepiece==0.1.99" "timm==0.9.10" "tokenizers==0.19.1" "transformers==4.40.1" "wandb==0.17.9" "tensorflow==2.15.0" "tensorflow_datasets==4.9.3" "tensorflow-metadata==1.15.0" "tensorflow_graphics==2021.12.3" "tensorflow-probability==0.23.0" "jax==0.4.23" "jaxlib==0.4.23" "imageio[ffmpeg]" "robosuite==1.4.1" "mujoco==2.3.7" bddl easydict cloudpickle gym "h5py" "opencv-python==4.9.0.80" "numba==0.59.1" "numpy==1.26.4"
"$UV" pip install --find-links "$PROJECT/wheels" --index-url https://mirrors.aliyun.com/pypi/simple --no-deps -e ./openvla -e ./LIBERO -e ./dlimp
python -c "import torch; print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0)); print(torch.ones(2,device='cuda').sum())"
"$UV" pip install --no-deps wheels/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
"$UV" pip freeze > results/environment.txt
date -Is > results/install_complete.txt
