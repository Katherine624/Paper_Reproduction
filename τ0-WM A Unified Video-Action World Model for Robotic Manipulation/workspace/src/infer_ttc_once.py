"""Run the official TTC policy on a few offline taco_play observations."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from infer_vam_once import PROJECT_ROOT, UPSTREAM_ROOT, save_observation, select_samples
from web_infer_utils.posttrain_taco_play_withTTC.TauPolicyTTC import TauPolicy


def build_policy_config(output_dir: Path, vam_checkpoint: Path, sim_checkpoint: Path) -> Path:
    source = UPSTREAM_ROOT / "configs/deployment/tau_posttrain_taco_play_abs_ttc.yaml"
    with source.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    wan_root = PROJECT_ROOT / "weights/wan2.2-ti2v-5b"
    config["diffusion_model"]["model_path"] = str(vam_checkpoint)
    if sim_checkpoint.is_dir() and not (
        sim_checkpoint / "diffusion_pytorch_model.bin.index.json"
    ).exists():
        sim_checkpoint = sim_checkpoint / "diffusion_pytorch_model.bin"
    config["sim_diffusion_model"]["model_path"] = str(sim_checkpoint)
    config["vae_path"] = str(wan_root / "Wan2.2_VAE.pth")
    config["text_encoder"]["checkpoint_path"] = str(
        wan_root / "models_t5_umt5-xxl-enc-bf16.pth"
    )
    config["text_encoder"]["tokenizer_path"] = str(wan_root / "google/umt5-xxl")
    stats = str(UPSTREAM_ROOT / "configs/data/stat_file/taco_play_abs.json")
    config["statistics_file"] = stats
    config["sim_statistics_file"] = stats
    for key in (
        "vae_class_path",
        "textenc_class_path",
        "diffusion_model_class_path",
        "pipeline_class_path",
        "sim_diffusion_model_class_path",
        "sim_pipeline_class_path",
    ):
        config[key] = str(UPSTREAM_ROOT / config[key])

    path = output_dir / "ttc_config.yaml"
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline official TTC inference")
    parser.add_argument("--vam-checkpoint", type=Path, required=True)
    parser.add_argument("--sim-checkpoint", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--proposals", type=int, default=2)
    parser.add_argument("--heldout-offset", type=int, default=300)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "runs/inference/ttc-public"
    )
    args = parser.parse_args()
    if args.proposals < 2:
        parser.error("TTC requires at least two action proposals")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, data_config = select_samples(args.count, args.heldout_offset)
    with Path(data_config["statistic_files"][0]).open(encoding="utf-8") as file:
        stats = json.load(file)

    policy = TauPolicy(
        config_file=str(
            build_policy_config(
                output_dir, args.vam_checkpoint.resolve(), args.sim_checkpoint.resolve()
            )
        ),
        device=torch.device("cuda:0"),
        attention_impl="sdpa",
        enable_self_attn_fused_qkv=True,
    )
    results = []
    for ordinal, (sample_index, sample) in enumerate(samples, start=1):
        sample_dir = output_dir / f"sample_{ordinal:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        obs = sample["video"][:, :, 0].permute(1, 0, 2, 3).contiguous()
        save_observation(obs, sample_dir / "observation.png")
        state = (
            sample["state"][0].numpy() * np.asarray(stats["state"]["std"])
            + np.asarray(stats["state"]["mean"])
        ).astype(np.float32)
        reference = (
            sample["actions"].numpy() * np.asarray(stats["action"]["std"])
            + np.asarray(stats["action"]["mean"])
        )
        policy.reset()
        if policy.add_mem:
            policy.memory["obs"].clear()
        actions = policy.play(
            obs=obs,
            prompt=sample["caption"],
            state=state,
            num_inference_steps=5,
            execution_step=len(reference),
            sample_solver="euler",
            shift=1.0,
            N_proposals=args.proposals,
            sim_refine=True,
            sim_num_inference_steps=5,
        )
        if actions.shape != reference.shape:
            raise ValueError(f"Action shape mismatch: {actions.shape} != {reference.shape}")
        mae = float(np.mean(np.abs(actions - reference)))
        with (sample_dir / "actions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["step"] + [f"pred_{i}" for i in range(actions.shape[1])]
                            + [f"reference_{i}" for i in range(reference.shape[1])])
            for step, (predicted, actual) in enumerate(zip(actions, reference, strict=True)):
                writer.writerow([step, *predicted.tolist(), *actual.tolist()])
        result = {
            "dataset_index": sample_index,
            "task": sample["caption"],
            "action_shape": list(actions.shape),
            "action_mae": mae,
            "proposals": args.proposals,
            "sim_refine": True,
            "note": "Offline TTC smoke test, not robot task success.",
        }
        with (sample_dir / "result.json").open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        results.append(result)
        print(f"TTC sample {ordinal}: index={sample_index}, MAE={mae:.4f}", flush=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
