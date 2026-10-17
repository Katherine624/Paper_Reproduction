# τ0-WM 小规模复现记录（2026-10-13）

## 已完成

- 在 GPUHome 单卡 RTX 6000D 84GB 上建立 Python 3.10 / PyTorch 2.7.1+cu128 环境，并准备官方代码、`lerobot/taco_play` 数据集、VAM / Simulator / Wan VAE / T5 权重。
- VAM：以官方预训练权重初始化，对 `taco_play` 的 7 维动作空间进行 30 步、30 个样本的短训练；保存 `runs/checkpoints/vam-gpuhome-30/step_30`。官方根权重的动作输入/输出头是 20 维，载入时这三个 7 维头参数重新初始化，其余参数载入。
- Simulator：以官方预训练权重初始化，进行 30 步、30 个样本的短训练；保存 `runs/checkpoints/sim-gpuhome-30/step_30`。官方模拟器的动作输入投影是 26 维，载入时该 7 维投影重新初始化，其余参数载入。
- VAM 离线推理：10 个未参与这次 30 样本训练的帧，输出 33×7 动作，平均动作 MAE 0.3928。见 `runs/inference/vam-gpuhome-30/summary.json`。
- Simulator 离线推理：3 个样本，每个分别输入数据集动作和 VAM 预测动作，得到 6 段预测视频与首/末帧对照图；两组最终帧平均 MSE 分别为 0.0833 和 0.0839。见 `runs/inference/sim-gpuhome-30-comparison/summary.json`。
- TTC 联合推理：VAM 生成 2 个候选，由 Simulator 辅助选择；1 个样本成功输出 33×7 动作，MAE 0.3393。两模型载入时均无缺失、尺寸不匹配或意外权重。见 `runs/inference/ttc-gpuhome-30/summary.json`。

## 结果如何解释

上述是训练环境、两个后训练流程和离线联合推理的**跑通验证**。只训练 30 步、30 个样本，预测视频存在明显伪影。测试帧与训练帧不同，但没有按 episode 严格划分，故 MAE/MSE 不代表泛化能力；单次 TTC 与 VAM 的 MAE 差异也不能证明 TTC 更优。官方公开的原始权重动作维度与 `taco_play` 不同，因此不能直接把原始权重的输出当作该任务的有效性能基线。尚未进行机器人/仿真环境的闭环任务成功率评测，也未完成多随机种子、24 任务×多轨迹的论文级评测。官方仓库的 README 提供训练和推理示例，未见相应闭环评测脚本与环境配置。

## 云端复现命令

工作目录：`/root/rivermind-data/tau0-wm-reproduction`

```bash
env/tau0/bin/python -u workspace/src/train_vam.py --init-checkpoint weights/vam --train-steps 30 --subset-size 30 --output-dir runs/checkpoints/vam-gpuhome-30
env/tau0/bin/python -u workspace/src/train_sim.py --init-checkpoint weights/simulator/diffusion_pytorch_model.bin --train-steps 30 --subset-size 30 --output-dir runs/checkpoints/sim-gpuhome-30
env/tau0/bin/python -u workspace/src/infer_vam_once.py --checkpoint runs/checkpoints/vam-gpuhome-30/step_30 --count 10 --heldout-offset 300 --output-dir runs/inference/vam-gpuhome-30
env/tau0/bin/python -u workspace/src/infer_sim_once.py --checkpoint runs/checkpoints/sim-gpuhome-30/step_30 --count 3 --heldout-offset 300 --compare-vam-dir runs/inference/vam-gpuhome-30 --output-dir runs/inference/sim-gpuhome-30-comparison
env/tau0/bin/python -u workspace/src/infer_ttc_once.py --vam-checkpoint runs/checkpoints/vam-gpuhome-30/step_30 --sim-checkpoint runs/checkpoints/sim-gpuhome-30/step_30 --count 1 --proposals 2 --heldout-offset 300 --output-dir runs/inference/ttc-gpuhome-30
```

日志位于 `runs/logs/`。云实例按小时收费；确认文件已下载后，须在 GPUHome 控制台关机以停止 GPU 计费。
