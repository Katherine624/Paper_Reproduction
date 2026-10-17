"""Build auditable result tables from the official evaluator's per-episode log."""
import argparse
import csv
import json
import re
import statistics
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--trials-per-task", type=int, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    logs = list((root / "logs").glob("*.txt"))
    if len(logs) != 1:
        raise RuntimeError(f"Expected one evaluator log, found {len(logs)}")
    content = logs[0].read_text(encoding="utf-8")
    episodes, groups = [], OrderedDict()
    task = None
    for line in content.splitlines():
        if line.startswith("Task: "):
            task = line.removeprefix("Task: ").strip()
            groups.setdefault(task, [])
        elif line in ("Success: True", "Success: False"):
            if task is None:
                raise RuntimeError("An outcome has no task description")
            success = line.endswith("True")
            groups[task].append(success)
            episodes.append(dict(task_id=len(groups)-1, task=task,
                                 episode=len(groups[task]), success=int(success)))
    rows = [dict(task_id=i, task=task, episodes=len(outcomes),
                 successes=sum(outcomes), failures=len(outcomes)-sum(outcomes),
                 success_rate=sum(outcomes)/len(outcomes) if outcomes else None)
            for i, (task, outcomes) in enumerate(groups.items())]
    for name, values in (("episodes.csv", episodes), ("per_task_results.csv", rows)):
        if values:
            with (root / name).open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    exit_file = root / "exit_code.txt"
    exit_code = int(exit_file.read_text().strip()) if exit_file.exists() else None
    errors = "Episode error:" in content or "Traceback (most recent call last)" in content
    expected_total = args.expected_tasks * args.trials_per_task
    total_matches = re.findall(r"^Total episodes: (\d+)$", content, re.M)
    success_matches = re.findall(r"^Total successes: (\d+)$", content, re.M)
    successes = sum(e["success"] for e in episodes)
    complete = (exit_code == 0 and not errors and len(rows) == args.expected_tasks
                and len(episodes) == expected_total
                and all(r["episodes"] == args.trials_per_task for r in rows)
                and total_matches == [str(expected_total)]
                and success_matches == [str(successes)])
    summary = dict(status="complete" if complete else "incomplete_or_failed",
                   model="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
                   checkpoint_revision="cb689ec0e3347c13667d70a78a3447388f5c3bb8",
                   seed=195, expected_episodes=expected_total,
                   completed_episodes=len(episodes), successes=successes,
                   overall_success_rate=successes/len(episodes) if episodes else None,
                   task_count=len(rows), exit_code=exit_code,
                   video_count=len(list((root / "videos").glob("*.mp4"))),
                   evaluator_log=str(logs[0].relative_to(root)), tasks=rows,
                   limitation="Reduced evaluation, one seed; not the paper's full protocol.")
    started_file, finished_file = root / "started_at.txt", root / "finished_at.txt"
    if started_file.exists() and finished_file.exists():
        started = datetime.fromisoformat(started_file.read_text().strip())
        finished = datetime.fromisoformat(finished_file.read_text().strip())
        summary["started_at"] = started.isoformat()
        summary["finished_at"] = finished.isoformat()
        summary["wall_time_seconds"] = (finished - started).total_seconds()
    query_times = [float(t) for t in re.findall(r"Action query time = ([0-9.]+) sec", content)]
    if query_times:
        summary["action_query_seconds"] = dict(
            count=len(query_times), mean=statistics.mean(query_times),
            median=statistics.median(query_times), maximum=max(query_times),
            note="Per action chunk of 16 actions; includes the first warm-up query.")
    gpu_samples = root / "gpu_samples.csv"
    if gpu_samples.exists():
        with gpu_samples.open(encoding="utf-8-sig", newline="") as f:
            values = [int(r["memory_used_mib"]) for r in csv.DictReader(f)]
        if values:
            summary["gpu_memory_samples"] = dict(
                count=len(values), maximum_mib=max(values),
                note="Total device memory sampled every 10 seconds; not an exact peak measurement.")
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        "Cosmos Policy：LIBERO 官方检查点评测记录",
        "",
        f"状态：{'完成' if complete else '未完成或发生错误，不能作为完整评测结果'}",
        f"任务范围：LIBERO-Spatial，{args.expected_tasks} 个任务，每任务 {args.trials_per_task} 个回合，seed=195。",
        "模型：作者发布的完整 Cosmos-Policy-LIBERO-Predict2-2B 权重；未加载我们训练的 LoRA。",
        f"已完成回合：{len(episodes)}/{expected_total}；成功：{successes}。",
        f"成功率：{successes / len(episodes):.1%}" if episodes else "成功率：无有效回合。",
        "成功判定：LIBERO 仿真环境返回的任务完成状态，不以 loss 或视频观感代替。",
        "",
        "各任务结果：",
    ]
    for row in rows:
        report.append(f"  task {row['task_id']}: {row['successes']}/{row['episodes']}  {row['task']}")
    if "wall_time_seconds" in summary:
        report.extend(["", f"总用时：{summary['wall_time_seconds'] / 60:.1f} 分钟（包括模型加载、环境初始化、评测与视频保存）。"])
    if query_times:
        report.append(f"每次预测 16 个动作的平均计算时间：{statistics.mean(query_times):.3f} 秒；中位数 {statistics.median(query_times):.3f} 秒。")
    if "gpu_memory_samples" in summary:
        report.append(f"显存采样最大值：{summary['gpu_memory_samples']['maximum_mib']} MiB（每 10 秒一次，不能代表精确瞬时峰值）。")
    report.extend([
        "",
        "运行条件与修改：",
        "  本机 RTX 5060 Laptop 8 GB；WSL2 Ubuntu；实际依赖版本见 environment.json。",
        "  加载阶段先在 CPU 内存初始化网络、转换为 BF16，再移到 GPU，降低初始化时的显存峰值。",
        "  模型架构、官方权重、图像输入处理及动作去噪步骤均保留；没有量化成 8 位或 4 位。",
        "  保留官方动作块长度 16、动作去噪 5 步；配置和完整命令见 command.txt。",
        "  加入独立视频保存目录、运行错误立即退出、每任务结束关闭仿真环境。",
        "  source_snapshot/ 与 source_hashes.json 保存关键运行源码与校验值。",
        "",
        "与论文完整评测的区别：",
        "  本次仅覆盖 Spatial 套件、一个随机种子、每任务 10 回合；没有覆盖其他 LIBERO 套件或 RoboCasa。",
        "  这是缩小评测，用于验证官方模型和训练/推理环境；不能宣称复现了论文完整成功率。",
        "  24 个任务的表述对应 RoboCasa 方案；本次 LIBERO-Spatial 是 10 个任务。",
        "",
        "已有训练证据（与本次官方模型评测分开）：",
        "  cloud-results/spatial_rtxpro6000_01/ 中有约 3 小时的单卡 LoRA 微调日志，日志记录到 step 1960。",
        "  latest_checkpoint.txt 指向 iter_000001900；exports/lora_iter_000001900.pt 为导出的适配器权重。",
        "  训练日志没有 Done with training 标记：应报告为按时间预算进行的短程训练，而非完整训练结束。",
        "  本次成功率衡量作者发布的模型，不衡量上述适配器的效果。",
        "",
        "结果文件：",
        "  summary.json：结构化汇总；per_task_results.csv：分任务结果；episodes.csv：逐回合结果。",
        "  logs/：官方评测日志；console.log：完整控制台输出；videos/：回放与预测画面对照视频。",
        "  exit_code.txt：程序退出状态；0 代表正常退出。",
        "",
        "来源：https://github.com/NVlabs/cosmos-policy/blob/main/LIBERO.md",
        "官方权重：https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
    ])
    (root / "RESULTS.txt").write_text("\n".join(report) + "\n", encoding="utf-8-sig")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
