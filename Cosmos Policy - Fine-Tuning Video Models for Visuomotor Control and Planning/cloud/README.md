# 单张 A100 80GB：LIBERO-Spatial 训练准备

这套代码使用 NVIDIA 官方 Cosmos Policy 2B 模型、Wan 2.1 tokenizer、LIBERO 数据加载器和 `HybridEDMSDE` 训练目标。我们只训练 LIBERO-Spatial 的 10 个任务，在 Transformer 的注意力和 MLP 层加入 LoRA。默认每卡 batch 2、梯度累积 8，有效 batch 16；训练循环预算 3.5 小时。它是学习完整微调流程的小规模实验，不等于论文的四套件全参数训练。

`scripts/train_libero_cloud.py` 有两个入口：`prepare` 只下载；`run` 默认只打印训练计划，加入 `--execute` 才训练。本地未启动过这套云端训练。

## 本项目文件

- `scripts/train_libero_cloud.py`：下载、检查、建立 Spatial 数据视图和启动训练。
- `cloud/cosmos_policy_cloud.patch`：对固定版本官方源码的三处改动：训练时间预算、LoRA 初始化与底座权重键名对应、从本地读取底座文件。
- `upstream/cosmos-policy/`：本地官方源码已应用上述改动。

## 云端准备

租用 Linux x86-64 的单张 A100 80GB。建议使用至少 64GB 内存、150GB 可持久化磁盘。数据集在初始化时会把示范解码到内存；Docker 镜像、环境、数据和完整训练检查点都会占用磁盘。实际显存与速度要以云机器为准。

在云端创建项目目录，将本地的 `scripts/train_libero_cloud.py` 和 `cloud/cosmos_policy_cloud.patch` 上传到对应目录，然后执行：

```bash
cd ~/cosmos-repro
git clone https://github.com/NVlabs/cosmos-policy.git upstream/cosmos-policy
git -C upstream/cosmos-policy checkout 18a2accadf4e7a3531e56754102af5a24d2316da
git -C upstream/cosmos-policy apply --check ../../cloud/cosmos_policy_cloud.patch
git -C upstream/cosmos-policy apply ../../cloud/cosmos_policy_cloud.patch
cd upstream/cosmos-policy
docker build -t cosmos-policy docker
```

项目目录应放在可持久化磁盘上。若云平台提供已配置 CUDA 12.8 的官方兼容镜像，也可直接在其中执行后面的 `uv sync`。NVIDIA 官方安装步骤见 [`SETUP.md`](../upstream/cosmos-policy/SETUP.md)。

从 `~/cosmos-repro/upstream/cosmos-policy` 启动容器：

```bash
docker run -u root \
  -e HOST_USER_ID=$(id -u) -e HOST_GROUP_ID=$(id -g) \
  -v "$HOME/.cache:/home/cosmos/.cache" \
  -v "$HOME/cosmos-repro:/workspace" \
  --gpus all --ipc=host -it --rm \
  -w /workspace/upstream/cosmos-policy \
  --entrypoint bash cosmos-policy
```

在容器内执行：

```bash
uv sync --extra cu128 --group libero --python 3.10
source .venv/bin/activate
hf auth login
python ../../scripts/train_libero_cloud.py prepare
python ../../scripts/train_libero_cloud.py run --run-name spatial_a100_01
```

Hugging Face 账号需要有 `nvidia/Cosmos-Predict2-2B-Video2World` 的访问权限；云端需要重新登录。`prepare` 固定数据和底座权重的 Hugging Face revision，只下载 Spatial 所需的 10 个示范文件、500 个 rollout 文件、T5 embedding、底座模型和 tokenizer。不要把访问令牌写入脚本。

检查打印出的命令、路径与 3.5 小时预算后，在容器内启动：

```bash
python ../../scripts/train_libero_cloud.py run --run-name spatial_a100_01 --hours 3.5 --execute
```

预算从官方训练器完成模型与检查点加载后开始计时，在一次梯度更新结束后停止，并沿用官方的最终保存流程。下载、Docker 构建、环境安装、数据解码、模型加载和最后一次保存都在训练预算之外，因此云机器总计费时间会超过 3.5 小时。每 100 次参数更新另存一次检查点。最终文件在 `cloud-runs/cosmos_policy_cloud/libero_spatial_lora/spatial_a100_01/checkpoints/`。

如果 batch 2 显存不足，换一个运行名并使用 batch 1、梯度累积 16：

```bash
python ../../scripts/train_libero_cloud.py run \
  --run-name spatial_a100_b1 --batch-size 1 --grad-accum 16 \
  --hours 3.5 --execute
```

相同运行名再次启动时，官方检查点机制会从 `latest_checkpoint.txt` 恢复。脚本拒绝在同名运行下修改训练命令。训练完成后先保存检查点和 `config.yaml`，再做 LIBERO 仿真评测；三四小时的损失下降不能直接当成论文的成功率结果。

## 一次参数更新做了什么

1. 官方数据集从成功示范与 rollout 取样，读入两路相机、机器人状态、动作和语言指令；T5 embedding 已预计算。
2. 官方 tokenizer 把图像、状态、动作和值等表示组织成 9 个潜在帧；条件帧保持可见。
3. `HybridEDMSDE` 对目标潜在帧采样噪声强度并加噪；视频 Transformer 在条件和噪声强度下预测干净潜在量，计算训练损失。
4. 梯度只更新注入 Transformer 的 LoRA 参数。累积 8 个 batch 后更新一次优化器和学习率调度器。
5. 按步数或时间预算保存模型、优化器、调度器与梯度缩放状态，供恢复和后续评测。

官方全规模配置与硬件说明见 [`LIBERO.md`](../upstream/cosmos-policy/LIBERO.md)。
