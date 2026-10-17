"""Write the human-readable report from verified experiment records."""
import json
import pathlib

root = pathlib.Path(__file__).resolve().parents[1]
read = lambda relative: json.loads((root / relative).read_text(encoding='utf-8'))
v = read('output/verification.json')
assert v['status'] == 'passed'
c = read('runs/walker_walk_seed0/config.json')
m = read('runs/walker_walk_seed0/run_manifest.json')
a = read('runs/walker_walk_seed0/training_audit.json')
s = read('output/walker_walk_eval10/summary.json')
r = read('output/runtime_probe/report.json')
metrics = [json.loads(x) for x in (root / 'runs/walker_walk_seed0/metrics.jsonl').read_text().splitlines() if x.strip()]
losses = [x for x in metrics if 'model_loss' in x]
model_updates = a['optimizer_iterations']['model']
report = f'''# Dreamer v1 第一阶段复现记录

已完成云端环境适配、真实环境短程训练、权重保存与独立评测。当前结论是**训练与推理流程跑通**；论文最终性能尚未复现。

## 目标与实施范围

目标是 [Dream to Control: Learning Behaviors by Latent Imagination](../input/dreamer-iclr2020.pdf)，ICLR 2020 的 Dreamer v1。按“训练能跑通、加载权重测性能、评测规模可缩减”的要求，先执行 Walker Walk 单任务、单训练种子的短程实验。

实现基于作者 [danijar/dreamer](https://github.com/danijar/dreamer)，固定提交 `56d4d444dfd0582b0e79dab80aebbea74c0ce40d`。目前核查官方主页、README 与 Releases 未找到可直接下载的匹配预训练权重，因此评测使用本次自行训练的权重。

## 实际运行结果

| 项目 | 实测结果 |
|---|---|
| 控制任务 | `{c['task']}` |
| 训练种子 | `{c['seed']}` |
| 训练环境步数 | {a['environment_steps']:,}，含 {c['prefill']:,} 步随机预采样 |
| 网络更新计数 | 世界模型、actor、value 各 {model_updates:,} 次，含作者初始化模型时的 1 次更新 |
| 输入 | 64 × 64 RGB；动作重复 {c['action_repeat']} 次 |
| 序列批次、想象长度 | {c['batch_size']} × {c['batch_length']}；H = {c['horizon']} |
| 数值精度 | float{c['precision']} |
| 训练入口耗时 | {m['elapsed_seconds'] / 60:.2f} 分钟，含采样、编译和周期评测 |
| 独立评测 | {s['episodes']} 回合，每回合 {c['time_limit']:,} 个原始环境步 |
| 评测平均回报 ± 标准差 | **{s['mean_return']:.2f} ± {s['std_return']:.2f}** |
| 回报范围 | {s['min_return']:.2f}–{s['max_return']:.2f} |
| 独立评测回合耗时 | {s['elapsed_seconds']:.2f} 秒，不含权重构建与加载 |
| 权重加载最大绝对误差 | {s['restore_max_abs_difference']} |
| 评测期间权重变化 | 无；encoder、RSSM 和 actor 哈希均保持一致 |

标准差使用样本标准差（ddof = 1），描述同一个 checkpoint 在不同评测回合间的波动，不能替代多个训练种子的方差。评测环境种子为 20000–20009，每回合重置环境与循环隐状态，关闭探索噪声；保留作者代码的随机 RSSM 和动作众数采样近似。

日志记录的世界模型损失由首个记录点的 {losses[0]['model_loss']:.2f} 变为最后记录点的 {losses[-1]['model_loss']:.2f}。所有 {len(losses)} 条损失与梯度记录均为有限值，代码对每次更新另有有限性断言。损失下降本身不能证明学会了稳定行走。

![训练与独立评测诊断图](figures/short_run_diagnostics.png)

## 验证与运行环境

- GPU 卷积、反向传播、EGL 渲染和完整 1,000 步环境回合通过。
- 手算 lambda-return（λ = 0、0.5、1）、零折扣边界与环境种子重复性测试通过。
- encoder、RSSM、decoder、reward、actor、value 六个模块的参数均发生变化。
- 独立评测入口只创建推理模块，不创建优化器、不读取训练 replay。
- 保存权重哈希、实际加载后的参数哈希、执行源码哈希与原始回报统计均经核对。

云端 GPU 为 NVIDIA RTX 6000D（85,651 MiB，计算能力 12.0），驱动 580.119.02。Python {r['python']}，TensorFlow {r['tensorflow']}，MuJoCo {r['mujoco']}，dm_control 1.0.48；完整版本见 [依赖锁定文件](../requirements.lock.txt)。

官方 TensorFlow 2.20 曾在该 GPU 上报 `CUDA_ERROR_INVALID_PTX`。本次使用经过 SHA256 校验的 [社区 sm_120 构建](https://github.com/Syraxius/TensorflowDockerBuilder/releases/tag/v2.21.0-sm120-cuda128-py310-py313)，不是 TensorFlow 官方发行包。构建来源、TensorFlow 源码提交和校验值见 [运行库来源](../runtime_sources.json)，实际测试见 [运行时检查](runtime_probe/report.json)。

## 解释边界与后续工作

1. 论文主实验覆盖 20 个控制任务，每任务 5,000,000 个环境步、5 个训练种子。本次仅单任务、单种子 {a['environment_steps']:,} 步，不能声称已经达到论文性能。师兄截图中的 Cosmos“24 个任务”是示例，未直接套用到 Dreamer。
2. 保留作者 TF2 简化实现的网络配置：400 单元，reward/value/actor 分别 2/3/4 个隐藏层；与论文附录的三层 300 单元配置存在差异。现代 MuJoCo/dm_control 与 2020 年环境也并不完全一致。
3. 已验证模型权重能加载并继续更新。但作者保存方式在当前 Keras 下未完整恢复优化器计数，随机状态与环境状态也不恢复，不能视为逐位一致的断点续训。本次正式短程训练连续完成。
4. 后续若获得与该实现、任务和结构匹配的可信预训练权重，可直接运行独立评测。若没有，则需要增加训练预算，并在扩展任务数前先完成一个任务的较长训练和性能核对。

## 文件与复查方式

- [原始评测回报 CSV](walker_walk_eval10/episodes.csv)、[评测汇总](walker_walk_eval10/summary.json)、[实际控制视频](walker_walk_eval10/episode_00.mp4)。
- [交叉核验结果](verification.json)、[基础数学与环境验证](core_validation.json)、[系统信息](system_info.json)。
- [训练配置](../runs/walker_walk_seed0/config.json)、[运行记录](../runs/walker_walk_seed0/run_manifest.json)、[参数更新审计](../runs/walker_walk_seed0/training_audit.json)。
- 本次推理 checkpoint 的 SHA256：`{s['checkpoint_sha256']}`。
- [运行说明与命令](../README.md)。云端项目目录：`/root/rivermind-data/dreamer-v1-repro`。
- `dreamer_v1_evidence.tar.gz` 备份包含代码、原始日志、权重和 replay；不含 Python 环境与 SSH 凭据。归档及逐文件哈希见同名 `.manifest.json`。
'''
(root / 'output/reproduction_report.md').write_text(report, encoding='utf-8')
print('Wrote output/reproduction_report.md from verified records.')
