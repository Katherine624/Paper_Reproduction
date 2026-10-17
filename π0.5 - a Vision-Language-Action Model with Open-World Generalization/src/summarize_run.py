"""Validate exported evidence and produce a concise Chinese reproduction record."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "output/reproduction-evidence"
TRAIN = EVIDENCE / "results/train-smoke"
EVAL = EVIDENCE / "results/eval-libero-spatial-seed7-trials10"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    trained = read_json(TRAIN / "training-verification.json")
    restored = read_json(TRAIN / "restore-verification.json")
    metrics = read_json(TRAIN / "metrics-summary.json")
    audit = read_json(TRAIN / "dataset-audit.json")
    evaluation = read_json(EVAL / "summary.json")
    episodes = [json.loads(x) for x in (EVAL / "episodes.jsonl").read_text().splitlines()]
    assert trained["training_steps"] == restored["restored_step"] == 100
    assert restored["parameter_samples_identical"] and trained["checkpoint_save_completed"]
    assert metrics["steps_with_finite_metrics"] == 100
    assert evaluation["completed"] and evaluation["episodes"] == len(episodes) == 100
    assert evaluation["successes"] == sum(int(x["success"]) for x in episodes)
    assert [x["episode_index"] for x in episodes] == list(range(100))
    assert len(evaluation["per_task"]) == 10
    assert all(x["episodes"] == 10 for x in evaluation["per_task"].values())
    assert not (EVAL / "infrastructure-error.txt").exists()
    for row in episodes:
        assert (EVAL / row["video"]).is_file(), row["video"]

    with (ROOT / "output/评测结果.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["任务编号", "任务", "成功次数", "尝试次数", "成功率", "随机种子"])
        for index, (task, values) in enumerate(evaluation["per_task"].items(), 1):
            writer.writerow([index, task, values["successes"], values["episodes"],
                             values["successes"] / values["episodes"], evaluation["seed"]])

    loss_rows = [json.loads(x) for x in (TRAIN / "metrics-from-log.jsonl").read_text().splitlines()]
    with (ROOT / "output/训练日志.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step_index", "loss", "grad_norm", "param_norm"])
        writer.writeheader()
        writer.writerows(loss_rows)

    checkpoint = trained["checkpoint_dir"] + "/" + str(trained["checkpoint_index"])
    report = f"""π0.5 复现记录：训练流程验证与公开权重仿真评测

本次按“训练跑通即可，推理使用公开 checkpoint 测性能”的要求执行。
训练使用 pi05_base；评测使用官方 pi05_libero，两部分分别记录。
这是 OpenPI 的短训练与缩减版 LIBERO 评测，不覆盖原论文的真实家庭开放世界实验。

服务器：NVIDIA L40S 48GB（nvidia-smi 显示 46068 MiB），24 核 CPU 配额、120 GiB 内存配额。
系统：Ubuntu 22.04.4 LTS；NVIDIA 驱动 575.57.08。
模型环境：Python 3.11.17、JAX 0.5.3；仿真环境：Python 3.8.20、MuJoCo 3.2.3、robosuite 1.4.1。
仿真画面由 OSMesa 在 CPU 上生成，模型计算使用 GPU。
官方代码：https://github.com/Physical-Intelligence/openpi
代码 commit：{evaluation['openpi_commit']}
服务器项目目录：/root/gpufree-data/pi05-repro

训练验证结果
- 使用官方 physical-intelligence/libero 数据集的 8 条真实示范，共 {audit['frames']} 帧。
- 数据版本：{audit['revision']}。
- 使用官方 LoRA 冻结规则，batch size=2，seed=42，共完成 100 次梯度更新。
- 全部 100 步的 loss、梯度范数和参数范数均为有限值，梯度范数均大于 0。
- 前 10 步平均 loss：{metrics['mean_loss_first10']:.4f}；后 10 步平均 loss：{metrics['mean_loss_last10']:.4f}。
  loss 来自官方训练日志，原始打印精度为小数点后四位。
- 抽样检查确认 LoRA 参数更新、冻结参数保持不变。
- checkpoint 保存完成，并在独立的新进程中重载；{restored['sampled_tensors_verified']} 组参数样本完全一致，恢复的步数为 100。
- checkpoint 路径：{checkpoint}
  文件夹编号 99 对应从 0 开始计数的第 100 次更新。
- 短训练用于验证训练流程，不能据此判断训练已经收敛。

公开权重评测结果
- 权重：官方 pi05_libero。
- 任务集：LIBERO Spatial，10 个任务；每个任务 10 次，共 100 次。
- seed=7；每个任务使用初始状态编号 0—9；每执行 5 步重新规划动作。
- 成功 {evaluation['successes']} 次，失败 {100-evaluation['successes']} 次，成功率 {evaluation['success_rate']:.1%}。
- 评测耗时约 {evaluation['elapsed_seconds']/60:.1f} 分钟（包含场景初始化、首次推理编译和视频保存）。
- 100 次尝试均有记录和回放视频；没有把通信或推理异常当作普通任务失败计数。
- 本次只测一个任务集且每任务 10 次，不等同于完整 LIBERO 评测规模。

操作顺序与复查命令
1. 连接服务器，在 /root/gpufree-data/pi05-repro 下工作。
2. 检查基础环境：bash scripts/check-model-env.sh 和 bash scripts/check-libero-env.sh。
3. 准备真实数据子集、官方基础权重，计算归一化统计。
   source env-model.sh
   openpi/.venv/bin/python scripts/pi05_experiment.py norms
4. 短训练与重载：bash scripts/run-train-smoke.sh。
   已完成的训练目录不会自动覆盖；再次训练需要另设实验目录，并隔离结果文件。
5. 复查训练结果：openpi/.venv/bin/python scripts/check-training-results.py。
6. 加载官方权重：bash scripts/run-policy-server.sh。
7. 在另一个终端评测：bash scripts/run-libero-evaluation.sh。
   评测目录已存在时脚本会拒绝覆盖；重跑应另设结果目录。

本地交付
- 评测结果.csv：10 个任务的成功次数与成功率。
- 训练日志.csv：100 步训练 loss 与梯度、参数范数。
- reproduction-evidence/：原始日志、配置、核验结果、脚本和 100 段视频。
- reproduction-evidence.tar.gz：上述证据包。

权重来源和下载完整性
两份 checkpoint 均来自官方 openpi-assets 存储桶。
已按官方对象清单检查文件大小，并使用 MD5 或 CRC32C 校验（部分大对象只提供 CRC32C）。
完整 checkpoint 保存在服务器；本地证据包包含结果和复现所需脚本，不包含大型模型权重。

收尾状态
实验进程已停止，GPU 显存已释放；云实例本身未关闭，仍会继续计费。
工具自动审批拦截了清理本地临时权重缓存的操作，仅返回“blocked by policy”，未提供更具体原因。
因此 data/official-checkpoints 和 work/checkpoint-chunks 中的本地缓存仍保留，不影响实验结果。
"""
    path = ROOT / "output/π0.5复现结果.txt"
    path.write_text(report, encoding="utf-8")
    print("REPORT_READY", path)
    print("EVALUATION", evaluation["successes"], "/", evaluation["episodes"])
