"""Run offline Simulator predictions on held-out taco_play samples."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = PROJECT_ROOT / "upstream" / "tau-0-wm"
sys.path.insert(0, str(UPSTREAM_ROOT))

from data.example_dataset_sim import CustomLeRobotDataset
from web_infer_utils.simulator.TauSimulator import TauSimulator


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def select_samples(count: int, heldout_offset: int):
    with (PROJECT_ROOT / "workspace/configs/taco_abs_joint.yaml").open(
        encoding="utf-8"
    ) as file:
        data_config = yaml.safe_load(file)["data"]
    data_config["data_roots"] = [str(PROJECT_ROOT / "data/raw/taco_play")]
    data_config["statistic_files"] = [
        str(UPSTREAM_ROOT / "configs/data/stat_file/taco_play_abs.json")
    ]
    dataset = CustomLeRobotDataset(
        **data_config,
        add_mem_cond=True,
        n_mem=3,
        return_reward=True,
    )
    permutation = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(3331459158),
    )
    samples = []
    for index in permutation[heldout_offset : heldout_offset + max(20, count * 10)]:
        try:
            samples.append((int(index), dataset.get_batch(int(index))))
            if len(samples) == count:
                return samples, data_config
        except Exception as error:
            print(f"Skipping unreadable sample {int(index)}: {error}", flush=True)
    raise RuntimeError(f"Found only {len(samples)} readable held-out samples")


def build_simulator_config(output_dir: Path, checkpoint: Path) -> Path:
    source = UPSTREAM_ROOT / "configs/deployment/tau_simulator_posttrain_taco_play_abs.yaml"
    with source.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    wan_root = PROJECT_ROOT / "weights/wan2.2-ti2v-5b"
    if checkpoint.is_dir() and not (
        checkpoint / "diffusion_pytorch_model.bin.index.json"
    ).exists():
        checkpoint = checkpoint / "diffusion_pytorch_model.bin"
    config["diffusion_model"]["model_path"] = str(checkpoint)
    config["vae_path"] = str(wan_root / "Wan2.2_VAE.pth")
    config["text_encoder"]["checkpoint_path"] = str(
        wan_root / "models_t5_umt5-xxl-enc-bf16.pth"
    )
    config["text_encoder"]["tokenizer_path"] = str(wan_root / "google/umt5-xxl")
    config["statistics_file"] = str(
        UPSTREAM_ROOT / "configs/data/stat_file/taco_play_abs.json"
    )
    for key in (
        "vae_class_path",
        "textenc_class_path",
        "diffusion_model_class_path",
        "diffusion_scheduler_class_path",
        "pipeline_class_path",
    ):
        config[key] = str(UPSTREAM_ROOT / config[key])

    config_path = output_dir / "simulator_config.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    return config_path


def frame_to_rgb(frame: torch.Tensor | np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    array = np.transpose(array, (1, 2, 0))
    return np.clip((array + 1.0) * 127.5, 0, 255).astype(np.uint8)


def save_comparison(obs: torch.Tensor, reference: torch.Tensor, predicted: np.ndarray, path: Path) -> None:
    rows = []
    for view in range(obs.shape[0]):
        rows.append(
            np.concatenate(
                [
                    frame_to_rgb(obs[view]),
                    frame_to_rgb(reference[view]),
                    frame_to_rgb(predicted[view]),
                ],
                axis=1,
            )
        )
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def load_predicted_actions(vam_dir: Path, sample_index: int) -> np.ndarray:
    with (vam_dir / "summary.json").open(encoding="utf-8") as file:
        vam_results = json.load(file)
    for ordinal, result in enumerate(vam_results, start=1):
        if result["dataset_index"] != sample_index:
            continue
        with (vam_dir / f"sample_{ordinal:02d}" / "actions.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            reader = csv.DictReader(file)
            columns = [name for name in reader.fieldnames if name.startswith("pred_")]
            actions = [[float(row[name]) for name in columns] for row in reader]
        return np.asarray(actions, dtype=np.float32)
    raise ValueError(f"No VAM prediction for dataset index {sample_index} in {vam_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Simulator on held-out samples")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--heldout-offset", type=int, default=300)
    parser.add_argument(
        "--actions-from", type=Path,
        help="VAM inference directory; use its predicted actions instead of dataset actions",
    )
    parser.add_argument(
        "--compare-vam-dir", type=Path,
        help="Run both dataset actions and VAM actions with one Simulator load",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "runs/inference/sim-continued300"
    )
    args = parser.parse_args()
    if args.actions_from is not None and args.compare_vam_dir is not None:
        parser.error("Use either --actions-from or --compare-vam-dir")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, data_config = select_samples(args.count, args.heldout_offset)

    with project_path(data_config["statistic_files"][0]).open(encoding="utf-8") as file:
        stats = json.load(file)
    simulator = TauSimulator(
        config_file=str(build_simulator_config(output_dir, args.checkpoint.resolve())),
        device=torch.device("cuda:0"),
    )
    results = []
    for ordinal, (sample_index, sample) in enumerate(samples, start=1):
        sample_dir = output_dir / f"sample_{ordinal:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        obs = sample["video"][:, :, 0].permute(1, 0, 2, 3).contiguous()
        reference_final = sample["video"][:, :, -1].permute(1, 0, 2, 3).contiguous()
        raw_actions = (
            sample["actions"].numpy() * np.asarray(stats["action"]["std"])
            + np.asarray(stats["action"]["mean"])
        ).astype(np.float32)
        action_options = [("dataset", raw_actions)]
        if args.actions_from is not None:
            action_options = [
                (
                    "vam_prediction",
                    load_predicted_actions(args.actions_from.resolve(), sample_index),
                )
            ]
        elif args.compare_vam_dir is not None:
            action_options.append(
                (
                    "vam_prediction",
                    load_predicted_actions(args.compare_vam_dir.resolve(), sample_index),
                )
            )

        for action_source, actions_to_simulate in action_options:
            result_dir = (
                sample_dir / action_source
                if args.compare_vam_dir is not None
                else sample_dir
            )
            result_dir.mkdir(parents=True, exist_ok=True)
            simulator.reset()
            memory = sample["mem"]
            for frame in range(memory.shape[2]):
                simulator.memory["obs"].append(
                    memory[:, :, frame : frame + 1].to(
                        device=simulator.device, dtype=simulator.dtype
                    )
                )
            predicted_final, predicted_reward = simulator.play(
                obs=obs,
                prompt=sample["caption"],
                actions=actions_to_simulate,
                num_inference_steps=5,
                execution_step=len(actions_to_simulate),
                sample_solver="unipc",
                shift=5.0,
                guide_scale=1,
                n_mem=3,
                save_sim_video=True,
                sim_path=str(result_dir / "predicted_video.mp4"),
            )
            reference_array = reference_final.numpy()
            if predicted_final.shape != reference_array.shape:
                raise ValueError(
                    f"Frame shape mismatch: predicted {predicted_final.shape}, "
                    f"reference {reference_array.shape}"
                )
            mse = float(np.mean((predicted_final - reference_array) ** 2))
            save_comparison(
                obs, reference_final, predicted_final, result_dir / "frame_comparison.png"
            )
            result = {
                "dataset_index": sample_index,
                "task": sample["caption"],
                "action_source": action_source,
                "final_frame_mse": mse,
                "predicted_reward": np.asarray(predicted_reward).tolist(),
                "comparison_columns": ["input", "reference_final", "predicted_final"],
                "comparison_rows": ["view_0", "view_1"],
                "inference_steps": 5,
                "note": "Offline example, not a benchmark score.",
            }
            with (result_dir / "result.json").open("w", encoding="utf-8") as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
            results.append(result)
            print(
                f"Simulator sample {ordinal} ({action_source}): "
                f"index={sample_index}, MSE={mse:.4f}",
                flush=True,
            )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
