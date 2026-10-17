# Environment record

Environment prepared and validated on 2026-09-18. No reproduction source code has been written, and no model weights or datasets have been downloaded.

## Host

- Windows with WSL2 Ubuntu.
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU.
- VRAM: 8,151 MiB.
- NVIDIA driver: 592.01.
- Driver-reported CUDA compatibility: 13.1.
- Project storage: D drive.

## Reproduction environment

- Python: 3.10.21.
- Virtual environment: `/home/xiao/.venvs/tau0-wm`.
- Package set: official `requirements.txt` from `sii-research/tau-0-wm`.
- Official Git checkout: `/mnt/d/xiao/Documents/ChatGPT/tao0/tau0-wm-reproduction/upstream/tau-0-wm`.
- User-space compiler/CUDA toolkit: `/home/xiao/.toolchains/tau0`.
- PyTorch: 2.7.1+cu128.
- Torchvision: 0.22.1+cu128.
- Torchaudio: 2.7.1+cu128.
- CUDA toolkit: 12.8.93.
- DeepSpeed: 0.15.3.
- Transformers: 4.57.6.
- Diffusers: 0.35.2.
- LeRobot: 0.4.0.
- xFormers: 0.0.31.post1.

The NVIDIA driver may report CUDA 13.1 while PyTorch uses its own CUDA 12.8 runtime. This is normal. CUDA 12.8 was selected because the RTX 5060 has compute capability 12.0; the original CUDA 12.6 wheel could see the card but could not launch a kernel.

## Start a shell for this project

Run these commands inside WSL Ubuntu:

```bash
source /home/xiao/.venvs/tau0-wm/bin/activate
export CUDA_HOME=/home/xiao/.toolchains/tau0
export PATH=/home/xiao/.venvs/tau0-wm/bin:/home/xiao/.toolchains/tau0/bin:/usr/bin:/bin
export CC=/home/xiao/.toolchains/tau0/bin/x86_64-conda-linux-gnu-cc
export CXX=/home/xiao/.toolchains/tau0/bin/x86_64-conda-linux-gnu-c++
export TRITON_CACHE_DIR=/home/xiao/.cache/triton-tau0
cd /mnt/d/xiao/Documents/ChatGPT/tao0/tau0-wm-reproduction/upstream/tau-0-wm
```

## Validation completed

- PyTorch detected the RTX 5060 and completed a CUDA matrix multiplication.
- DeepSpeed imported successfully with the CUDA 12.8 toolkit.
- Transformers, Diffusers, LeRobot, Decord, OpenCV, xFormers, and TorchAO imported successfully.
- The official `configs/tau_model/posttrain_taco_play_abs.yaml` file parsed successfully.
- xFormers CUTLASS and Flash-Attention 2 execution paths completed a GPU attention operation.

On this Blackwell GPU, xFormers' automatic dispatcher currently chooses Flash-Attention 3, which fails with `invalid argument`. For later local experiments, use the official SDPA option or explicitly avoid FA3. We will handle that only when we begin writing/running code.

## Local compatibility notes

- `packaging==25.0` and `opencv-python-headless==4.12.0.88` are local compatibility pins required by LeRobot 0.4.0.
- `evdev` is intentionally omitted because WSL lacks the kernel input headers needed to build it without a system package install. It is an optional Linux keyboard/input dependency of `pynput`, not part of the core world-model path.
- `uv pip check` reports Decord as built for another platform, but `decord==0.6.0` imports correctly in this WSL environment.

## Hardware scope

The local 8 GB GPU is appropriate for:

- dependency and CUDA validation;
- reading configurations and inspecting tensor contracts;
- dataset conversion and small data-pipeline tests;
- CPU or reduced-size component tests;
- carefully selected quantized/offloaded inference experiments.

It is not sufficient for reproducing the full 5B/5.5B training run. Full model training and practical full-precision inference require a multi-GPU server or a substantially larger accelerator.

## Deliberately deferred

- Model weights have not been downloaded.
- Training data has not been downloaded.
- No source code, training script, or experiment configuration has been authored.
- No official YAML path has been modified.
