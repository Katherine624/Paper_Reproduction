# τ0-WM: A Unified Video-Action World Model for Robotic Manipulation

本目录归档这篇论文的**小规模复现**：我们编写的训练与推理脚本、配置、日志、离线预测结果，以及两份自己训练的 checkpoint。它不是论文性能结果的完整复现。

## 内容

- `workspace/src/`：VAM、Simulator 的训练与离线推理，以及 TTC 联合推理脚本。
- `workspace/configs/`：`taco_play` 示例配置。
- `runs/logs/`：训练与推理日志。
- `runs/inference/`：动作 CSV、指标 JSON、预测视频和对照图。
- `runs/checkpoints/`：checkpoint 的小型配置和索引文件；大型权重分块保存在同仓库的 [Release 下载区](https://github.com/Katherine624/Paper_Reproduction/releases/tag/tau0-wm-2026-10-17)。
- `runs/reproduction_status_2026-10-13.md`：实验过程、命令、结果和限制。
- `checkpoint_manifest.json`：Release 分块清单及 SHA-256 校验值。
- `restore_checkpoints.py`：下载、拼接并校验 checkpoint。

官方源码来自 [`sii-research/tau-0-wm`](https://github.com/sii-research/tau-0-wm)，本次使用的 commit 为 `a3e4c0fe58b88974ab6d9b1522af97e5e46be04d`。官方预训练权重、Wan VAE/T5 和公开 `lerobot/taco_play` 数据集没有重复上传；请从各自官方仓库获取。这里的代码需要与上述官方源码一起使用。

## 本次实际运行

在一张 RTX 6000D 84GB 上，VAM 与 Simulator 各用 30 个样本训练 30 步。VAM 在 10 个离线样本上的动作 MAE 均值为 0.3928；Simulator 生成了 3 个样本的两组动作条件视频；TTC 用 2 个候选完成了 1 个样本的联合推理。预测视频仍有明显伪影，样本也没有按 episode 严格隔离，因此这些数值**不能作为任务成功率或论文基准成绩**。

本机另有一份较早的 `vam-taco-play-subset100/step_100` 文件夹，但只剩第 1 权重分片，缺少第 2 分片和索引，无法作为完整 checkpoint 恢复，故未放入 Release。

## 恢复权重

在仓库根目录执行：

```bash
python "τ0-WM A Unified Video-Action World Model for Robotic Manipulation/restore_checkpoints.py"
```

脚本将从同仓库 Release 下载分块，校验并重建 `runs/checkpoints/vam-gpuhome-30/step_30/` 和 `runs/checkpoints/sim-gpuhome-30/step_30/` 下的两份权重文件。下载总量约 22 GB，请预留足够空间。
