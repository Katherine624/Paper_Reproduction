"""Real-data pi05 LoRA smoke run using the upstream training implementation.

Only configuration, explicit episode selection and verification hooks are added.
The official model, loss, optimizer update and checkpoint implementation are used.
"""
import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("PI05_ROOT", "/root/gpufree-data/pi05-repro"))
sys.path.insert(0, str(ROOT / "openpi"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import flax.traverse_util as traverse
import jax
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from openpi import transforms
from openpi.training import checkpoints, config, data_loader, optimizer, sharding, weight_loaders
from scripts import compute_norm_stats, train

EPISODES = list(range(8))
REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
RESULTS = ROOT / "results" / "train-smoke"
RESULTS.mkdir(parents=True, exist_ok=True)
ORIGINAL_DATASET = data_loader.create_torch_dataset


def subset_dataset(data_config, action_horizon, model_config):
    if data_config.repo_id != "physical-intelligence/libero":
        return ORIGINAL_DATASET(data_config, action_horizon, model_config)
    root = ROOT / "data" / "libero-subset"
    meta = LeRobotDatasetMetadata(data_config.repo_id, root=root, revision=REVISION)
    dataset = LeRobotDataset(
        data_config.repo_id, root=root, episodes=EPISODES, revision=REVISION,
        delta_timestamps={key: [t / meta.fps for t in range(action_horizon)]
                          for key in data_config.action_sequence_keys},
        video_backend="pyav",
    )
    expected = sum(meta.episodes[i]["length"] for i in EPISODES)
    assert len(dataset) == expected
    sample = dataset[0]
    for key in ["state", "actions"]:
        assert np.isfinite(np.asarray(sample[key])).all()
    audit = {"repo": data_config.repo_id, "revision": REVISION, "episodes": EPISODES,
             "frames": len(dataset), "fps": meta.fps,
             "selected_tasks": sorted({t for i in EPISODES for t in meta.episodes[i]["tasks"]}),
             "sample_shapes": {k: list(v.shape) for k, v in sample.items() if hasattr(v, "shape")}}
    (RESULTS / "dataset-audit.json").write_text(json.dumps(audit, indent=2))
    print("REAL_DATA_READY", len(dataset), "frames", flush=True)
    if data_config.prompt_from_task:
        dataset = data_loader.TransformedDataset(dataset, [transforms.PromptFromLeRobotTask(meta.tasks)])
    return dataset


def make_config(exp_name):
    base = config.get_config("pi05_libero")
    model = dataclasses.replace(base.model, paligemma_variant="gemma_2b_lora",
                                action_expert_variant="gemma_300m_lora")
    return dataclasses.replace(
        base, name="pi05_libero_lora_smoke", exp_name=exp_name, model=model,
        freeze_filter=model.get_freeze_filter(), ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(ROOT / "checkpoints/pi05_base/params")),
        assets_base_dir=str(ROOT / "assets"), checkpoint_base_dir=str(ROOT / "checkpoints/training"),
        batch_size=2, num_workers=0, num_train_steps=100, seed=42,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=10, peak_lr=5e-5, decay_steps=100, decay_lr=5e-6),
        save_interval=100, log_interval=1, keep_period=None, wandb_enabled=False,
    )


def flat_params(state):
    return traverse.flatten_dict(state.params.to_pure_dict())


def sample_arrays(flat, keys):
    result = {}
    for i, key in enumerate(keys):
        value = flat[tuple(key)].reshape(-1)
        indices = np.linspace(0, value.size - 1, min(256, value.size), dtype=int)
        result[f"p{i}"] = np.asarray(jax.device_get(value[indices]), dtype=np.float32)
    return result


def train_run(cfg):
    original_init, original_save, original_log = train.init_train_state, checkpoints.save_state, train.wandb.log
    baseline = {}

    def checked_init(*args, **kwargs):
        state, partition = original_init(*args, **kwargs)
        flat = flat_params(state)
        trainable = traverse.flatten_dict(state.params.filter(cfg.trainable_filter).to_pure_dict())
        frozen = traverse.flatten_dict(state.params.filter(cfg.freeze_filter).to_pure_dict())
        lora_keys = sorted(k for k in trainable if "lora" in "/".join(map(str, k)))
        assert lora_keys
        # Sample every LoRA tensor and several frozen tensors; retain exact values.
        keys = lora_keys + sorted(frozen)[:3]
        initial = sample_arrays(flat, keys)
        np.savez(RESULTS / "initial-parameter-samples.npz", **initial)
        baseline.update(keys=keys, initial=initial, lora_count=len(lora_keys))
        metadata = {"keys": [list(k) for k in keys], "lora_sampled_tensors": len(lora_keys),
                    "trainable_parameters": sum(int(x.size) for x in trainable.values()),
                    "total_parameters": sum(int(x.size) for x in flat.values())}
        (RESULTS / "parameter-audit.json").write_text(json.dumps(metadata, indent=2))
        print("PARAMETER_AUDIT", metadata["trainable_parameters"], "trainable", flush=True)
        return state, partition

    def checked_log(data, *args, **kwargs):
        if "loss" in data:
            metrics = {k: float(np.asarray(v)) for k, v in data.items()}
            assert all(np.isfinite(v) for v in metrics.values()), metrics
            metrics["step_index"] = kwargs.get("step")
            with (RESULTS / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(metrics) + "\n")
        return original_log(data, *args, **kwargs)

    def checked_save(manager, state, loader, step):
        final = sample_arrays(flat_params(state), baseline["keys"])
        changes = {k: float(np.max(np.abs(final[k] - baseline["initial"][k]))) for k in final}
        n = baseline["lora_count"]
        assert any(changes[f"p{i}"] > 0 for i in range(n)), "No sampled LoRA parameter changed"
        assert all(changes[f"p{i}"] == 0 for i in range(n, len(changes))), "Frozen parameter changed"
        assert all(np.isfinite(v).all() for v in final.values())
        np.savez(RESULTS / "final-parameter-samples.npz", **final)
        report = {"training_steps": int(state.step), "checkpoint_index": step,
                  "checkpoint_dir": str(cfg.checkpoint_dir), "sampled_lora_updated": True,
                  "sampled_frozen_unchanged": True, "max_sampled_parameter_change": max(changes.values())}
        original_save(manager, state, loader, step)
        manager.wait_until_finished()
        report["checkpoint_save_completed"] = True
        (RESULTS / "training-verification.json").write_text(json.dumps(report, indent=2))
        print("TRAINING_VERIFIED", json.dumps(report), flush=True)

    train.init_train_state, checkpoints.save_state, train.wandb.log = checked_init, checked_save, checked_log
    train.main(cfg)


def restore_run(cfg):
    audit = json.loads((RESULTS / "parameter-audit.json").read_text())
    expected = np.load(RESULTS / "final-parameter-samples.npz")
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    _, init_rng = jax.random.split(jax.random.key(cfg.seed))
    shape, _ = train.init_train_state(cfg, init_rng, mesh, resume=True)
    manager, _ = checkpoints.initialize_checkpoint_dir(cfg.checkpoint_dir, keep_period=None, overwrite=False, resume=True)
    state = checkpoints.restore_state(manager, shape, None)
    actual = sample_arrays(flat_params(state), audit["keys"])
    for key, value in actual.items():
        np.testing.assert_array_equal(value, expected[key])
    assert int(state.step) == cfg.num_train_steps
    report = {"restored_step": int(state.step), "sampled_tensors_verified": len(actual),
              "parameter_samples_identical": True, "restored_in_fresh_process": True}
    (RESULTS / "restore-verification.json").write_text(json.dumps(report, indent=2))
    manager.close()
    print("RESTORE_VERIFIED", json.dumps(report), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["norms", "train", "restore"])
    parser.add_argument("--exp-name", default="real8_steps100_seed42")
    args = parser.parse_args()
    cfg = make_config(args.exp_name)
    data_loader.create_torch_dataset = subset_dataset
    (RESULTS / "training-config.txt").write_text(repr(cfg))
    if args.mode == "norms":
        original_get = config.get_config
        config.get_config = lambda name: cfg if name == cfg.name else original_get(name)
        compute_norm_stats.main(cfg.name)
    elif args.mode == "train":
        train_run(cfg)
    else:
        restore_run(cfg)
