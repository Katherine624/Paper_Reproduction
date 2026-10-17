"""Prepare and launch a short, single-A100 Cosmos Policy LIBERO-Spatial run.

This launcher uses NVIDIA's actual 2B policy model and training loop. Nothing
trains unless ``run --execute`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = PROJECT_ROOT / "upstream" / "cosmos-policy"
DATASET_ID = "nvidia/LIBERO-Cosmos-Policy"
DATASET_REVISION = "b463cbd5d114909e4daea65eb03bed9787290042"
BASE_MODEL_ID = "nvidia/Cosmos-Predict2-2B-Video2World"
BASE_MODEL_REVISION = "f50c09f5d8ab133a90cac3f4886a6471e9ba3f18"
MODEL_FILE = "model-480p-16fps.pt"
TOKENIZER_FILE = "tokenizer/tokenizer.pth"
SPATIAL_ROLLOUT_GLOB = "episode_data--suite=libero_spatial--*.hdf5"


def project_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    project = Path(args.project_root).expanduser().resolve()
    dataset = project / "datasets" / "LIBERO-Cosmos-Policy"
    model = project / "checkpoints" / "Cosmos-Predict2-2B-Video2World"
    return project, dataset, model


def check_assets(dataset: Path, model: Path) -> tuple[list[Path], list[Path]]:
    demos = sorted((dataset / "success_only" / "libero_spatial_regen").glob("*.hdf5"))
    rollouts = sorted((dataset / "all_episodes").glob(SPATIAL_ROLLOUT_GLOB))
    required = [
        dataset / "success_only" / "t5_embeddings.pkl",
        model / MODEL_FILE,
        model / TOKENIZER_FILE,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if len(demos) != 10 or len(rollouts) != 500 or missing:
        raise RuntimeError(
            "LIBERO-Spatial 文件不完整；先运行 prepare。"
            f" 示范文件={len(demos)}/10，rollout 文件={len(rollouts)}/500，缺失={missing}"
        )
    return demos, rollouts


def prepare(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    project, dataset, model = project_paths(args)
    dataset.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    print("下载 LIBERO-Spatial 的 10 个示范任务、500 个 rollout 和文本 embedding…", flush=True)
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=dataset,
        allow_patterns=[
            "success_only/libero_spatial_regen/*.hdf5",
            "success_only/t5_embeddings.pkl",
            f"all_episodes/{SPATIAL_ROLLOUT_GLOB}",
        ],
        max_workers=8,
    )
    for filename in (MODEL_FILE, TOKENIZER_FILE):
        print(f"下载预训练文件：{filename}", flush=True)
        hf_hub_download(
            repo_id=BASE_MODEL_ID,
            revision=BASE_MODEL_REVISION,
            filename=filename,
            local_dir=model,
        )
    demos, rollouts = check_assets(dataset, model)
    print(f"准备完成：{len(demos)} 个示范 HDF5，{len(rollouts)} 个 rollout HDF5。")
    print(f"数据：{dataset}\n底座权重：{model}")


def link_once(source: Path, target: Path) -> None:
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"已有链接指向其他数据：{target}")
        return
    if target.exists():
        raise RuntimeError(f"目标已存在，请检查：{target}")
    target.symlink_to(source, target_is_directory=source.is_dir())


def create_spatial_view(dataset: Path, output_root: Path, rollouts: list[Path]) -> Path:
    """Expose only Spatial data to the official loader without copying HDF5 files."""
    view = output_root / "data-view" / "LIBERO-Cosmos-Policy"
    success = view / "success_only"
    rollout_view = view / "all_episodes"
    success.mkdir(parents=True, exist_ok=True)
    rollout_view.mkdir(parents=True, exist_ok=True)
    link_once(dataset / "success_only" / "libero_spatial_regen", success / "libero_spatial_regen")
    link_once(dataset / "success_only" / "t5_embeddings.pkl", success / "t5_embeddings.pkl")
    for source in rollouts:
        link_once(source, rollout_view / source.name)
    # The official loader writes Spatial-only normalization statistics here.
    # Do not import the original statistics, which cover all four suites.
    return view


def make_command(args: argparse.Namespace, view: Path, model: Path) -> list[str]:
    overrides = [
        "experiment=cosmos_predict2_2b_480p_libero",
        "model.config.fsdp_shard_size=1",
        "model.config.use_lora=true",
        f"model.config.lora_rank={args.lora_rank}",
        f"model.config.lora_alpha={args.lora_rank}",
        f"checkpoint.load_path={model / MODEL_FILE}",
        f"model.config.tokenizer.vae_pth={model / TOKENIZER_FILE}",
        f"dataloader_train.dataset.data_dir={view / 'success_only'}",
        f"dataloader_train.dataset.rollout_data_dir={view / 'all_episodes'}",
        f"dataloader_train.dataset.t5_text_embeddings_path={view / 'success_only' / 't5_embeddings.pkl'}",
        f"dataloader_train.batch_size={args.batch_size}",
        f"dataloader_train.num_workers={args.workers}",
        f"trainer.grad_accum_iter={args.grad_accum}",
        "trainer.max_iter=100000",
        "trainer.logging_iter=10",
        "trainer.run_validation=false",
        "scheduler.warm_up_steps=[20,0]",
        "scheduler.cycle_lengths=[1000,100000000000000]",
        f"checkpoint.save_iter={args.save_every}",
        "job.project=cosmos_policy_cloud",
        "job.group=libero_spatial_lora",
        f"job.name={args.run_name}",
        "job.wandb_mode=disabled",
    ]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        "-m",
        "cosmos_policy.scripts.train",
        "--config=cosmos_policy/config/config.py",
        "--",
        *overrides,
    ]


def run(args: argparse.Namespace) -> None:
    if args.execute and os.name != "posix":
        raise RuntimeError("训练入口须在 Linux 云机器或 Linux Docker 容器中运行。")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        raise ValueError("run-name 只能包含英文字母、数字、下划线和连字符")
    if args.hours <= 0 or args.batch_size <= 0 or args.grad_accum <= 0 or args.lora_rank <= 0:
        raise ValueError("hours、batch-size、grad-accum 和 lora-rank 必须为正数")
    if args.workers < 0 or args.save_every <= 0:
        raise ValueError("workers 必须非负，save-every 必须为正数")

    project, dataset, model = project_paths(args)
    upstream = project / "upstream" / "cosmos-policy"
    trainer_file = upstream / "cosmos_policy" / "trainer.py"
    model_file = upstream / "cosmos_policy" / "_src" / "predict2" / "models" / "text2world_model.py"
    config_file = upstream / "cosmos_policy" / "config" / "experiment" / "cosmos_policy_experiment_configs.py"
    if not trainer_file.is_file() or not model_file.is_file() or not config_file.is_file():
        raise RuntimeError(f"找不到官方源码：{upstream}")
    if "COSMOS_POLICY_TRAIN_BUDGET_SECONDS" not in trainer_file.read_text():
        raise RuntimeError("官方 trainer 尚未应用 cloud/cosmos_policy_cloud.patch")
    if "Mapped {remapped} base checkpoint tensors" not in model_file.read_text():
        raise RuntimeError("官方模型尚未应用 cloud/cosmos_policy_cloud.patch")
    if "COSMOS_POLICY_BASE_MODEL_PATH" not in config_file.read_text():
        raise RuntimeError("官方配置尚未应用 cloud/cosmos_policy_cloud.patch")
    _, rollouts = check_assets(dataset, model)
    output_root = Path(args.output_root).expanduser().resolve()
    view = output_root / "data-view" / "LIBERO-Cosmos-Policy"
    command = make_command(args, view, model)
    run_dir = output_root / "cosmos_policy_cloud" / "libero_spatial_lora" / args.run_name
    budget_seconds = round(args.hours * 3600)
    manifest = {
        "dataset_revision": DATASET_REVISION,
        "base_model_revision": BASE_MODEL_REVISION,
        "training_budget_seconds": budget_seconds,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "checkpoint_directory": str(run_dir / "checkpoints"),
        "command": command,
    }
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    if not args.execute:
        print("仅显示计划；添加 --execute 才会创建数据视图并启动训练。")
        return

    create_spatial_view(dataset, output_root, rollouts)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "cloud_run.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["command"] != command:
            raise RuntimeError(f"同名运行的配置不同，请换一个 run-name：{run_dir}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["BASE_DATASETS_DIR"] = str(view.parent)
    env["IMAGINAIRE_OUTPUT_ROOT"] = str(output_root)
    env["COSMOS_POLICY_TRAIN_BUDGET_SECONDS"] = str(budget_seconds)
    env["COSMOS_POLICY_BASE_MODEL_PATH"] = str(model / MODEL_FILE)
    env["COSMOS_POLICY_SKIP_UNUSED_CHECKPOINT_DOWNLOADS"] = "1"
    env["WANDB_MODE"] = "disabled"
    print(f"开始训练，命令：{shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=upstream, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="包含 scripts/upstream/datasets 的项目目录")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("prepare", help="从 Hugging Face 下载 Spatial 数据和底座权重，不训练")
    run_parser = subparsers.add_parser("run", help="显示或执行单卡 LoRA 训练")
    run_parser.add_argument("--run-name", required=True)
    run_parser.add_argument("--output-root", default=str(PROJECT_ROOT / "cloud-runs"))
    run_parser.add_argument("--hours", type=float, default=3.5, help="训练循环预算，不含下载和初始化")
    run_parser.add_argument("--batch-size", type=int, default=2)
    run_parser.add_argument("--grad-accum", type=int, default=8)
    run_parser.add_argument("--lora-rank", type=int, default=32)
    run_parser.add_argument("--workers", type=int, default=4)
    run_parser.add_argument("--save-every", type=int, default=100)
    run_parser.add_argument("--execute", action="store_true", help="实际启动训练")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
