"""Run offline VAM predictions on held-out taco_play samples."""

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

from data.example_dataset import CustomLeRobotDataset
from web_infer_utils.posttrain_taco_play.TauPolicy import TauPolicy


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
    dataset = CustomLeRobotDataset(**data_config)
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


def build_policy_config(output_dir: Path, checkpoint: Path) -> Path:
    source = UPSTREAM_ROOT / "configs/deployment/tau_posttrain_taco_play_abs.yaml"
    with source.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    wan_root = PROJECT_ROOT / "weights/wan2.2-ti2v-5b"
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
        "pipeline_class_path",
    ):
        config[key] = str(UPSTREAM_ROOT / config[key])

    config_path = output_dir / "policy_config.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    return config_path


def save_observation(obs: torch.Tensor, destination: Path) -> None:
    views = []
    for view in obs:
        pixels = view.permute(1, 2, 0).numpy()
        views.append(np.clip((pixels + 1.0) * 127.5, 0, 255).astype(np.uint8))
    Image.fromarray(np.concatenate(views, axis=1)).save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VAM on held-out samples")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--heldout-offset", type=int, default=300)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "runs/inference/vam-continued300"
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, data_config = select_samples(args.count, args.heldout_offset)

    with project_path(data_config["statistic_files"][0]).open(encoding="utf-8") as file:
        stats = json.load(file)

    policy = TauPolicy(
        config_file=str(build_policy_config(output_dir, args.checkpoint.resolve())),
        device=torch.device("cuda:0"),
        attention_impl="sdpa",
    )
    results = []
    for ordinal, (sample_index, sample) in enumerate(samples, start=1):
        sample_dir = output_dir / f"sample_{ordinal:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        obs = sample["video"][:, :, 0].permute(1, 0, 2, 3).contiguous()
        save_observation(obs, sample_dir / "observation.png")
        state_raw = (
            sample["state"][0].numpy() * np.asarray(stats["state"]["std"])
            + np.asarray(stats["state"]["mean"])
        ).astype(np.float32)
        true_actions = (
            sample["actions"].numpy() * np.asarray(stats["action"]["std"])
            + np.asarray(stats["action"]["mean"])
        )
        predicted_actions = policy.play(
            obs=obs,
            prompt=sample["caption"],
            state=state_raw,
            num_inference_steps=5,
            execution_step=len(true_actions),
            sample_solver="euler",
            shift=1.0,
        )
        if predicted_actions.shape != true_actions.shape:
            raise ValueError(
                f"Action shape mismatch: predicted {predicted_actions.shape}, "
                f"reference {true_actions.shape}"
            )
        mae = float(np.mean(np.abs(predicted_actions - true_actions)))
        with (sample_dir / "actions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["step"]
                + [f"pred_{axis}" for axis in range(predicted_actions.shape[1])]
                + [f"reference_{axis}" for axis in range(true_actions.shape[1])]
            )
            for step, (predicted, reference) in enumerate(
                zip(predicted_actions, true_actions, strict=True)
            ):
                writer.writerow([step, *predicted.tolist(), *reference.tolist()])
        result = {
            "dataset_index": sample_index,
            "task": sample["caption"],
            "action_shape": list(predicted_actions.shape),
            "action_mae": mae,
            "inference_steps": 5,
            "note": "Held-out example, not a benchmark score.",
        }
        with (sample_dir / "result.json").open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        results.append(result)
        print(f"VAM sample {ordinal}: index={sample_index}, MAE={mae:.4f}", flush=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
