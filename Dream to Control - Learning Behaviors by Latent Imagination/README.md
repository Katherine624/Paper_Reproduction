> **完整下载与恢复：[ARCHIVE.md](ARCHIVE.md)**。大文件保存在本仓库 Release。

# Dreamer v1 短程复现

目标论文：*Dream to Control: Learning Behaviors by Latent Imagination*，ICLR 2020，arXiv:1912.01603。

本项目按“训练跑通、加载权重评测、保存证据”的范围执行。20,000 个环境步的训练用于验证流程；不等同于论文每任务 5,000,000 步、5 个训练种子的性能复现。

本次实测结果、曲线、验证记录和后续工作见 [第一阶段复现报告](output/reproduction_report.md)。

## 实现与环境

- 作者代码：[danijar/dreamer](https://github.com/danijar/dreamer)，提交 `56d4d444dfd0582b0e79dab80aebbea74c0ce40d`。
- `src/dreamer/` 保留作者实现及 MIT 许可证，兼容性修改单独记录。
- 云端目录：`/root/rivermind-data/dreamer-v1-repro`。
- 云端实际 GPU：NVIDIA RTX 6000D，85,651 MiB，计算能力 12.0；驱动 580.119.02。
- 首次官方 TensorFlow 2.20 GPU 测试失败，错误为 `CUDA_ERROR_INVALID_PTX`。已改用社区编译的 TensorFlow 2.21.0+selfbuild（原生 sm_120），并通过 GPU 卷积、梯度与 MuJoCo EGL 渲染测试；来源及 SHA256 见 `runtime_sources.json`。
- 实验定义与验收标准见 `reproduction_spec.json`。

## 云端运行

以下命令都在云服务器终端执行。当前实例的 `.venv` 已安装并验证通过。迁移到同类 sm_120 GPU 的 Linux x86_64、Python 3.10 机器时，先安装 `libegl1 libgl1 libgles2`，再执行 `bash scripts/setup_remote.sh`。`requirements.lock.txt` 固定了本次实际使用的依赖；TensorFlow wheel 使用公开下载地址和 SHA256 校验。该 wheel 是社区构建，不是 TensorFlow 官方发行包；其他 GPU 架构需要重新选择并验证构建。

已安装环境的检查命令：

```bash
cd /root/rivermind-data/dreamer-v1-repro
source scripts/env.sh
python scripts/runtime_probe.py
python scripts/validate_core.py
```

短程训练保持作者 TF2 代码的网络规模、50×50 序列批次和 15 步想象长度，采用 float32：

```bash
python scripts/train.py \
  --logdir runs/walker_walk_seed0 \
  --task dmc_walker_walk --seed 0 \
  --steps 20000 --eval_every 5000 \
  --precision 32 --log_images False
```

独立评测从保存的配置构建 encoder、RSSM 和 actor，不创建优化器，不读取训练 replay buffer。每个回合重置环境与循环状态，使用不同环境种子：

```bash
python scripts/evaluate.py \
  --checkpoint runs/walker_walk_seed0 \
  --output output/walker_walk_eval10 \
  --episodes 10 --seed 20000 --video-episodes 1
python scripts/plot_results.py \
  --training runs/walker_walk_seed0 \
  --evaluation output/walker_walk_eval10 \
  --output output/figures
```

评测输出目录必须是新目录，防止混合两次运行的原始记录。

继续训练时使用相同 `--logdir` 和结构配置，并提高 `--steps`。训练入口会加载 `variables.pkl`，并使用原目录中的 replay 数据。已验证可以继续采样和更新，但当前作者保存方式在现代 Keras 下未恢复优化器计数，不能视为完整、逐位一致的断点续训；随机状态与环境状态也不恢复。本次正式短程实验使用连续运行。该 pickle 仅用于本项目自行生成的可信文件。

## 验收证据

| 文件 | 用途 |
|---|---|
| `output/runtime_probe/report.json` | GPU 卷积与梯度、1000 步控制环境和渲染检查 |
| `output/core_validation.json` | 手算 lambda-return、边界、环境种子测试 |
| `runs/.../config.json` | 实际配置 |
| `runs/.../run_manifest.json`、`run_history.jsonl` | 运行状态、源码哈希、版本与耗时 |
| `runs/.../metrics.jsonl` | 训练损失和逐回合回报 |
| `runs/.../training_audit.json` | 参数变化、优化器更新次数、checkpoint 哈希 |
| `runs/.../variables.pkl` | 作者格式的模型变量；继续训练还需 replay，存在上述恢复限制 |
| `runs/.../policy_weights.npz`、`.json` | 独立推理权重及校验元数据 |
| `output/.../episodes.csv`、`episodes.jsonl` | 每个评测回合的种子、回报和长度 |
| `output/.../summary.json` | 评测汇总、权重加载误差和参数不变性检查 |
| `output/.../episode_00.mp4` | 实际控制轨迹 |

## 与论文的差异及解释边界

1. 作者的 TF2 简化实现与论文附录 A 部分结构不同：这里保留 TF2 的 400 单元配置和 reward/value/actor 的 2/3/4 个隐藏层；论文描述三层 300 单元网络。
2. 使用现代 TensorFlow、MuJoCo 和 dm_control。运行完成后冻结实际依赖，现代物理环境的结果不能视为与 2020 年环境完全一致。
3. 使用 1 个训练种子和短程训练；评测 10 个回合的标准差反映环境与策略采样波动，不是训练种子间方差。
4. 保留作者的随机 RSSM 和通过 100 个样本近似动作分布众数的实现；评测关闭额外探索噪声，但策略并非完全确定性。
5. 初步核查官方主页、README 与 Releases 尚未发现可下载的官方预训练权重。自产短程权重只证明流程可运行，不能声称已经复现论文最终性能。
6. 兼容性补丁包括现代 mixed-precision/分布式接口、NumPy 别名替换、环境和 replay 种子固定，以及阻止评测调用触发训练。
7. 固定了 TensorFlow、NumPy、环境与 replay 的种子，但未验证跨机器或重复训练的逐位一致性。

## 结果复核与备份

完成训练、评测后，使用无需导入 TensorFlow 的脚本核对原始回报统计、源码和权重哈希、训练参数变化与推理期间参数不变性：

```bash
python scripts/verify_results.py \
  --training runs/walker_walk_seed0 \
  --evaluation output/walker_walk_eval10 \
  --output output/verification.json
```

`output/dreamer_v1_evidence.tar.gz` 的证据包包含代码、配置、原始日志、权重、replay 数据、评测记录与视频；不含 `.venv`、下载缓存或 SSH 凭据。解压到新目录后，可按前述依赖安装步骤重建运行环境。归档校验值和逐文件清单在同名 `.manifest.json` 中。

原始论文保存在 `input/`，文件校验值见 `input/source_manifest.json`。密码不保存在项目配置中。
