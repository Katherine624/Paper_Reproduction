# Cosmos Policy 复现准备

本目录用于复现论文 *Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning*。论文副本在 `reference/cosmos-policy-paper.pdf`。初始源码、环境、数据和权重已准备；本机教学模型已训练和评测。2026-09-22 已在云端 RTX PRO 6000 上进行约 3 小时的 2B 模型 LoRA 短程微调；2026-10-13 已在本机完成作者官方权重的 LIBERO-Spatial 缩小评测。具体结果见 [`RESULTS.txt`](eval-results/official_spatial_10x10_seed195/RESULTS.txt)。

## 当前复现结果（2026-10-13）

- 训练已跑通：云端短程 LoRA 微调日志记录至 step 1960，已保存并导出 iter 1900 适配器。日志没有完整训练结束标记，不能称为论文规模训练完成。
- 官方模型评测已跑通：LIBERO-Spatial 的 10 个任务，每任务 10 个回合，seed=195，共 100 回合，成功 99 回合，成功率 **99.0%**。task 5 为 9/10，其余任务均为 10/10。
- 评测使用作者完整 `Cosmos-Policy-LIBERO-Predict2-2B` 权重，没有加载我们训练的 LoRA。该成绩不代表我们适配器的效果。
- 全程在本机 RTX 5060 Laptop 上运行，总用时 67.7 分钟；程序正常退出。保存了 200 个视频（100 个回放和 100 个预测画面对照视频）、逐回合与逐任务结果、命令和环境记录。
- 结果目录：`eval-results/official_spatial_10x10_seed195/`。本次仅覆盖 Spatial、一个种子和每任务 10 回合；不是论文完整四套件、多种子评测。
- 为避免初始化阶段的显存峰值，网络先在 CPU 内存初始化并转换为 BF16，再放入 GPU；没有改变模型架构或量化成 8/4 位。每 10 秒采样的显存最大值为 5151 MiB，这不等于精确瞬时峰值。

再次运行可在 WSL 中执行以下命令；使用新的运行名称保存结果：

```bash
bash /mnt/d/xiao/Documents/ChatGPT/cosmos/scripts/eval_official_libero.sh libero_spatial 10 -1 official_spatial_repeat
```

## 目录

| 路径 | 内容 |
| --- | --- |
| `reference/` | 论文副本与来源信息 |
| `upstream/cosmos-policy/` | [NVIDIA 官方实现](https://github.com/NVlabs/cosmos-policy)，固定到 `18a2accadf4e7a3531e56754102af5a24d2316da` |
| `upstream/cosmos-policy/.venv/` | WSL/Linux 下 Python 3.10 虚拟环境 |
| `datasets/LIBERO-Cosmos-Policy/` | [NVIDIA 提供的训练数据](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy)，版本 `b463cbd5d114909e4daea65eb03bed9787290042` |
| `checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/` | [LIBERO 参考权重](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B)，版本 `cb689ec0e3347c13667d70a78a3447388f5c3bb8` |
| `checkpoints/Cosmos-Predict2-2B-Video2World/` | 训练初始化需要的底座权重和 Wan tokenizer，版本 `f50c09f5d8ab133a90cac3f4886a6471e9ba3f18`；与 Hugging Face 缓存使用硬链接，共享磁盘空间 |
| `env/` | 下载缓存和环境说明 |
| `runs/` | 后续实验输出 |

## 环境

官方要求 Linux x86-64、CUDA 12.8 容器。此电脑使用 WSL2 Ubuntu 和 NVIDIA GPU，依赖按官方 `uv.lock` 在 Linux 虚拟环境中安装。源码目录中的 `pyproject.toml` 和 `uv.lock` 保持原样。日后进入环境可在 WSL 中执行：

```bash
cd /mnt/d/xiao/Documents/ChatGPT/cosmos/upstream/cosmos-policy
source .venv/bin/activate
export BASE_DATASETS_DIR=/mnt/d/xiao/Documents/ChatGPT/cosmos/datasets
export HF_HOME=/mnt/d/xiao/Documents/ChatGPT/cosmos/env/hf-cache
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
```

本机 GPU 是 RTX 5060 Laptop，显存约 8 GB。官方报告 LIBERO 推理需约 6.8 GB 显存，而论文训练使用 64 张 80 GB H100。本机已通过上述加载调整运行完整官方模型的推理与评测；这不意味着 8 GB 可以容纳同规模训练。本机之前对约 4M 参数的教学模型训练 1000 步，并在 5 个仿真回合得到 0/5；这与当前官方 2B 模型的 99/100 结果是两项独立实验。当前采用 WSL 本地环境，未安装官方 Docker 镜像。

## 初始化模型

官方 LIBERO 训练配置引用以下两个文件，来自受限访问的 [Cosmos-Predict2-2B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World)：

- `model-480p-16fps.pt`（约 3.91 GB）
- `tokenizer/tokenizer.pth`（约 0.51 GB）

模型条款已由用户账户接受。文件同时位于 `env/hf-cache/hub/`，供官方代码按 `hf://` 路径读取。访问令牌保存在 WSL 用户目录，不在项目文件内。

## 已完成的静态校验

- LIBERO 数据集：2,050 个文件，总计 27,569,481,893 字节；与 Hugging Face 版本 `b463cbd5d114909e4daea65eb03bed9787290042` 的文件清单逐项比对，零缺失、零大小不符。
- 其中训练配置读取的 `success_only/`：48 个文件，总计 14,836,172,876 字节。
- LIBERO 参考权重 `Cosmos-Policy-LIBERO-Predict2-2B.pt`：SHA-256 为 `8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2`，与发布方元数据一致。
- 底座权重 `model-480p-16fps.pt`：3,913,017,214 字节，SHA-256 为 `fbc4f05d948078539cb5d7a8e59b6f40f940e4836b5b8a31dcab03e3e807a6f0`。Wan tokenizer：507,609,880 字节，SHA-256 为 `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`。两者均与发布方元数据一致。
- 环境：Python 3.10.21、PyTorch 2.7.0+cu128；可识别 RTX 5060 Laptop GPU。已完成教学模型训练和评测，以及上述官方 2B 权重的 100 回合评测；实际环境版本保存在结果目录的 `environment.json`。
- 已在离线模式下调用官方 `resolve_checkpoint_path`，确认训练配置使用的两个 `hf://` 路径都能命中本地缓存。

## 来源

- [论文项目主页](https://research.nvidia.com/labs/cosmos-lab/cosmos-policy/)
- [官方安装说明](https://github.com/NVlabs/cosmos-policy/blob/main/SETUP.md)
- [官方 LIBERO 说明](https://github.com/NVlabs/cosmos-policy/blob/main/LIBERO.md)
