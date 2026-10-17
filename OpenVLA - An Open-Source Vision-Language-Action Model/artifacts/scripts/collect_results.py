"""Validate completed evidence and create a compact, credential-free delivery bundle."""
import json, math, shutil, tarfile, hashlib, re, collections
from pathlib import Path
ROOT=Path("/root/gpufree-data/openvla-repro")
def read(p): return json.loads(p.read_text())
def select_run(folder, episodes):
    candidates=[]
    for p in folder.glob("*.summary.json"):
        s=read(p)
        if s["episodes"]==episodes:candidates.append((p,s))
    assert len(candidates)==1, candidates
    p,s=candidates[0]
    rows=[json.loads(x) for x in p.with_name(p.name.replace(".summary.json",".jsonl")).read_text().splitlines()]
    assert len(rows)==episodes and sum(x["success"] for x in rows)==s["successes"]
    assert not any(x["error"] is not None for x in rows)
    return p,s,rows
assert (ROOT/"results/pipeline_complete.txt").is_file()
ep,ev,rows=select_run(ROOT/"results/eval-spatial-100",100)
_,smoke,_=select_run(ROOT/"results/eval-smoke",3)
for task in range(10):
    rr=[x for x in rows if x["task_id"]==task]
    assert len(rr)==10 and sorted(x["initial_state_id"] for x in rr)==list(range(10))
trainroot=ROOT/"results/train-smoke"
train=read(trainroot/"summary.json")
resume=read(trainroot/"resume_summary.json")
metrics=[json.loads(x) for x in (trainroot/"metrics.jsonl").read_text().splitlines()]
rmetrics=[json.loads(x) for x in (trainroot/"resume_metrics.jsonl").read_text().splitlines()]
assert [x["step"] for x in metrics]==list(range(1,101))
assert [x["step"] for x in rmetrics]==list(range(101,106))
assert all(math.isfinite(x["loss"]) and math.isfinite(x["grad_norm"]) and x["grad_norm"]>0 for x in metrics+rmetrics)
assert train["max_parameter_update"]>0 and resume["max_parameter_update"]>0
assert resume["start_step"]==100 and resume["end_step"]==105 and resume["resumed_optimizer"]
assert all(len(s["generated_action"])==7 and all(math.isfinite(x) for x in s["generated_action"]) for s in [train,resume])
for step in [50,100,105]:
    p=trainroot/f"checkpoint-{step}"
    assert (p/"adapter_model.safetensors").stat().st_size>0 and (p/"training_state.pt").stat().st_size>0
taskstats=[{"task_id":i,"task":next(x["task"] for x in rows if x["task_id"]==i),"episodes":10,"successes":sum(x["success"] for x in rows if x["task_id"]==i)} for i in range(10)]
summary={"status":"passed","evaluation":ev,"training":train,"resume":resume,"per_task":taskstats,"training_first_loss":metrics[0]["loss"],"training_last_loss":metrics[-1]["loss"],"training_first_10_mean_loss":sum(x["loss"] for x in metrics[:10])/10,"training_last_10_mean_loss":sum(x["loss"] for x in metrics[-10:])/10}
(ROOT/"results/verified_summary.json").write_text(json.dumps(summary,indent=2))
out=ROOT/"delivery";out.mkdir(exist_ok=True)
for folder in ["results","logs","scripts"]:
    for p in (ROOT/folder).rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix in [".pt",".safetensors"]:continue
        target=out/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
for name in ["env.sh","source_manifest.json","wheel_manifest.json","extra_wheel_manifest.json"]:
    if (ROOT/name).exists():shutil.copy2(ROOT/name,out/name)
video_root=ROOT/"openvla/rollouts"
videos=list(video_root.rglob("*.mp4"))
timestamp=ep.name.removeprefix("EVAL-libero_spatial-openvla-").removesuffix("--spatial100.summary.json")
formal=[p for p in videos if p.name.startswith(timestamp)]
assert len(formal)==100, len(formal)
video_manifest=[{"path":str(p),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(formal)]
(out/"video_manifest.json").write_text(json.dumps(video_manifest,indent=2))
(out/"videos").mkdir(exist_ok=True)
for success,label in [(True,"success"),(False,"failure")]:
    candidates=sorted(p for p in formal if f"--success={success}--" in p.name)
    if candidates:shutil.copy2(candidates[0],out/"videos"/f"example_{label}.mp4")
versions={x:(ROOT/f"results/{x}_commit.txt").read_text().strip() for x in ["openvla","libero","dlimp"]}
config=read(trainroot/"config.json")
table="\n".join(f"| {s['task_id']} | {s['successes']}/10 | {s['task']} |" for s in taskstats)
report=f"""# OpenVLA 当晚复现记录

日期：2026-10-13。全部数字来自同目录的原始日志与 JSON/JSONL，已通过 collect_results.py 一致性检查。

## 完成情况

- 环境、GPU、无界面仿真及真实训练数据检查通过。
- 官方 Spatial 权重：10 个任务 × 10 个初始状态，共 **{ev['successes']}/100 成功（{ev['success_rate']:.1%}）**；没有被脚本捕获的运行异常。另有独立预检 {smoke['successes']}/3。
- 官方基础权重的 LoRA 短训：完成 **100 个 optimizer steps**，在第 50、100 步保存 checkpoint。
- 新进程读取第 100 步 adapter 和 optimizer state，继续完成第 101—105 步并保存 checkpoint；两次训练后均生成了有限的 7 维动作。
- 100 条正式评测视频均保留在服务器；本地交付包包含成功/失败示例（若存在）。

## 评测口径与结果

使用 openvla/openvla-7b-finetuned-libero-spatial，权重未经本轮短训修改。全局 seed=7；保留官方 get_libero_env 中 env.seed(0)，每个任务使用官方初始状态索引 0—9。保留官方 180° 图像旋转、面积比例 0.9 的中心裁剪、动作反归一化、夹爪变换、10 步等待和 220 步 Spatial 上限。BF16 + FlashAttention 2，无量化。

| 任务 ID | 成功次数 | 任务 |
|---|---:|---|
{table}

推理延迟中位数 {ev['action_latency_p50_s']:.3f} 秒，P95 {ev['action_latency_p95_s']:.3f} 秒；统计 {ev['timed_actions']} 次动作调用，排除前 10 次预热，计时包含图像处理和模型动作生成并同步 CUDA，不含 env.step。峰值 PyTorch allocated 显存 {ev['peak_gpu_allocated_bytes']/2**30:.2f} GiB（不是 nvidia-smi 总显存）。

论文附录 E、表 12 的 Spatial 结果为 84.7±0.9%，采用三个随机种子、每种子 500 次，共 1,500 次。本次使用公开单个 checkpoint 和 100 次固定初始状态评测，不能视为完整复现论文统计量，也不能据此宣称超过论文。

## 训练检查

基础模型：openvla/openvla-7b。公开数据：modified_libero_rlds 中的 libero_spatial_no_noops。使用官方模型、RLDSDataset、RLDSBatchTransform、ActionTokenizer、collator 和交叉熵损失；短训入口为 scripts/train_smoke.py。

LoRA rank=32、alpha=16、dropout=0、all-linear；batch={config['batch_size']}，AdamW，lr=5e-4，100 次 optimizer 更新，无梯度累积，开启官方图像增强。可训练参数 {config['trainable_parameters']:,}。

- 首步 loss={metrics[0]['loss']:.4f}，第 100 步 loss={metrics[-1]['loss']:.4f}；前/后 10 步均值 {summary['training_first_10_mean_loss']:.4f}/{summary['training_last_10_mean_loss']:.4f}。来自随机增强训练批次，不是测试集指标。
- 每步 loss、梯度均有限且梯度非零；被检查的 LoRA 参数最大变化 {train['max_parameter_update']:.6g}。
- 训练步耗时中位数 {train['median_step_seconds']:.3f} 秒；峰值 PyTorch allocated 显存 {train['peak_gpu_allocated_bytes']/2**30:.2f} GiB。
- checkpoint 重载后完成 5 次更新，参数继续变化，动作生成通过；其验证 loss={resume['verification_loss']:.4f}。该 loss 使用本轮首个训练批次，仅检查数值有效。
- 恢复了 optimizer/global_step 和 Python、NumPy、PyTorch RNG。RLDS 数据迭代器重新开始，**不声称精确延续样本顺序**。
- 100 步只验证训练管线，不证明收敛；正式成功率来自官方 Spatial 权重，不是这个短训 adapter。

## 环境与版本

Ubuntu 22.04，单张 NVIDIA L40S 46068 MiB，24 CPU 配额、120 GiB 内存配额。Python 3.10.16，Torch 2.2.0+cu121，Transformers 4.40.1，TensorFlow 2.15.0，FlashAttention 2.5.5，robosuite 1.4.1，MuJoCo 2.3.7。完整环境在 results/environment.txt，依赖一致性检查在 results/dependency_check.txt。

- OpenVLA: {versions['openvla']}
- LIBERO: {versions['libero']}（从实例已有官方源码缓存克隆到独立目录后固定此提交）
- dlimp: {versions['dlimp']}

Hugging Face 模型、数据的精确 revision、文件大小及 LFS SHA256 在 source_manifest.json；下载器在校验通过后才重命名为正式文件。评测代码补丁在 results/evaluation.patch，仅增加子集选择、记录与计时，并使用显式注册的官方本地模型类；TensorFlow 禁用 GPU，以免与 PyTorch 抢显存。安装时将 Protobuf 固定为 3.20.3 解决 Python 3.10 下的依赖约束。

## 文件与重跑

服务器根目录：/root/gpufree-data/openvla-repro。

- results/eval-spatial-100：逐条结果、配置和汇总。
- results/train-smoke：全部训练指标、训练配置、checkpoint-50/100/105。
- openvla/rollouts/2026_10_13：所有原始视频；交付包 video_manifest.json 记录正式 100 条视频的路径与校验值。
- scripts/run_pipeline.sh：实际运行参数；scripts/train_smoke.py：短训与恢复入口。
- results/verified_summary.json：一致性验证后的机器可读汇总。
- videos/example_success.mp4、videos/example_failure.mp4：示例轨迹。

权重、原始训练数据和 checkpoint 二进制保留在服务器，没有重复下载到本机。重跑时使用新输出目录，避免覆盖本次证据。先 source /root/gpufree-data/openvla-repro/env.sh，再按 REPRODUCE.txt 的命令执行。

来源：用户提供的 OpenVLA PDF 第 36—37 页；https://github.com/openvla/openvla；https://huggingface.co/openvla/openvla-7b；https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial；https://huggingface.co/datasets/openvla/modified_libero_rlds。
"""
(out/"复现报告.md").write_text(report,encoding="utf-8")
commands="""source /root/gpufree-data/openvla-repro/env.sh
cd "$OPENVLA_ROOT/openvla"
# Choose a new directory each time.
RUN="$OPENVLA_ROOT/results/recheck-$(date +%Y%m%d-%H%M%S)"
python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint "$OPENVLA_ROOT/models/spatial" --task_suite_name libero_spatial --num_trials_per_task 10 --center_crop True --seed 7 --local_log_dir "$RUN/eval" --run_id_note recheck
python "$OPENVLA_ROOT/scripts/train_smoke.py" --model "$OPENVLA_ROOT/models/base" --data "$OPENVLA_ROOT/data" --out "$RUN/train" --steps 100 --batch-size 2
python "$OPENVLA_ROOT/scripts/train_smoke.py" --model "$OPENVLA_ROOT/models/base" --data "$OPENVLA_ROOT/data" --out "$RUN/train" --steps 105 --batch-size 2 --resume "$RUN/train/checkpoint-100"
"""
(out/"REPRODUCE.txt").write_text(commands)
with tarfile.open(ROOT/"openvla_delivery.tar.gz","w:gz") as tar:tar.add(out,arcname="openvla_delivery")
print(json.dumps({"status":"passed","successes":ev["successes"],"episodes":100,"training_steps":100,"resume_steps":5,"bundle":str(ROOT/"openvla_delivery.tar.gz"),"bytes":(ROOT/"openvla_delivery.tar.gz").stat().st_size}),flush=True)
