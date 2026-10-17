"""Run the locally trained teaching model on LIBERO-Spatial task 0."""
"""
载入训练权重
→ 创建 LIBERO 仿真环境
→ 取得当前两张相机图片和机器人状态
→ 从随机噪声逐步生成 16 步动作
→ 执行动作
→ 取得新观测，再生成下一段动作
→ 记录成功率
"""
import argparse
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_libero_mini import (
    ACTION_DIM,
    CHUNK_SIZE,
    DATA_ROOT,
    FRAME_DIM,
    IMAGE_SIZE,
    NUM_FRAMES,
    PROJECT_ROOT,
    SIGMA_MAX,
    SIGMA_MIN,
    TEXT_PATH,
    MiniDenoiser,
    condition_mask,
    image_to_frame,
    normalize,
    vector_to_frame,
)

os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / "env" / "libero-config"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the small policy on LIBERO-Spatial task 0")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "runs" / "libero_mini" / "best.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--denoising-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "libero_mini" / "eval_task0.json")
    args = parser.parse_args()
    if args.episodes < 1 or args.denoising_steps < 2 or args.max_steps < 1:
        parser.error("--episodes and --max-steps must be positive; --denoising-steps must be at least 2")
    return args


def prepare_image(raw_image, device):
    flipped = np.ascontiguousarray(np.flipud(raw_image))
    image = Image.fromarray(flipped).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    values = np.array(image, dtype=np.float32, copy=True)
    tensor = torch.from_numpy(values).permute(2, 0, 1) / 127.5 - 1.0
    return tensor.unsqueeze(0).to(device)


def current_latents(observation, stats, device):
    wrist = prepare_image(observation["robot0_eye_in_hand_image"], device)
    primary = prepare_image(observation["agentview_image"], device)
    proprio = np.concatenate(
        (
            observation["robot0_gripper_qpos"],
            observation["robot0_eef_pos"],
            observation["robot0_eef_quat"],
        )
    ).astype(np.float32)
    proprio = torch.from_numpy(normalize(proprio, stats, "proprio")).unsqueeze(0).to(device)

    blank = wrist.new_zeros((1, FRAME_DIM))
    known = [
        blank,
        vector_to_frame(proprio).flatten(start_dim=1),
        image_to_frame(wrist).flatten(start_dim=1),
        image_to_frame(primary).flatten(start_dim=1),
    ]
    sequence = wrist.new_zeros((1, NUM_FRAMES, FRAME_DIM))
    sequence[:, :4] = torch.stack(known, dim=1)
    return sequence


def decode_actions(action_frame, stats):
    num_values = CHUNK_SIZE * ACTION_DIM
    indices = torch.arange(FRAME_DIM, device=action_frame.device) % num_values
    totals = action_frame.new_zeros((action_frame.shape[0], num_values))
    totals.scatter_add_(1, indices.expand(action_frame.shape[0], -1), action_frame)
    counts = torch.bincount(indices, minlength=num_values).clamp_min(1)
    normalized = (totals / counts).reshape(-1, CHUNK_SIZE, ACTION_DIM)
    normalized = normalized[0].clamp(-1.0, 1.0).cpu().numpy()

    minimum = np.asarray(stats["actions_min"], dtype=np.float32)
    maximum = np.asarray(stats["actions_max"], dtype=np.float32)
    actions = (normalized + 1.0) * 0.5 * (maximum - minimum) + minimum
    actions[:, -1] = np.where(actions[:, -1] >= 0.0, 1.0, -1.0)
    return actions


@torch.inference_mode()
def predict_action_chunk(model, observation, text, stats, device, denoising_steps):
    known_values = current_latents(observation, stats, device)
    known_mask = condition_mask("policy", 1, device)
    sample = torch.randn_like(known_values) * SIGMA_MAX
    sample = torch.where(known_mask[:, :, None], known_values, sample)
    schedule = torch.linspace(SIGMA_MAX, SIGMA_MIN, denoising_steps, device=device)

    for index, sigma_value in enumerate(schedule):
        sigma = sigma_value.expand(1)
        predicted_clean = model(sample, sigma, text, known_mask)
        if index + 1 == len(schedule):
            sample = predicted_clean
        else:
            next_sigma = schedule[index + 1]
            derivative = (sample - predicted_clean) / sigma_value
            sample = sample + (next_sigma - sigma_value) * derivative
        sample = torch.where(known_mask[:, :, None], known_values, sample)

    return decode_actions(sample[:, 4], stats)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = MiniDenoiser().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    stats = checkpoint["dataset_stats"]

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    if task.language != checkpoint["task_description"]:
        raise ValueError("The checkpoint was trained for a different LIBERO task")
    with TEXT_PATH.open("rb") as file:
        cached_text = pickle.load(file)
    text = cached_text[task.language].squeeze(0).float().mean(dim=0).unsqueeze(0).to(device)

    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file), camera_heights=256, camera_widths=256
    )
    env.seed(0)
    initial_states = suite.get_task_init_states(0)
    episode_results = []
    try:
        for episode_index in range(args.episodes):
            env.reset()
            observation = env.set_init_state(initial_states[episode_index % len(initial_states)])
            for _ in range(10):
                observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

            success = False
            executed_steps = 0
            while executed_steps < args.max_steps and not success:
                actions = predict_action_chunk(
                    model, observation, text, stats, device, args.denoising_steps
                )
                for action in actions:
                    if executed_steps >= args.max_steps:
                        break
                    observation, _, done, _ = env.step(action.tolist())
                    executed_steps += 1
                    if done:
                        success = True
                        break

            episode_results.append({"episode": episode_index, "success": success, "steps": executed_steps})
            print(f"episode={episode_index} success={success} steps={executed_steps}", flush=True)
    finally:
        env.close()

    successes = sum(result["success"] for result in episode_results)
    result = {
        "task": task.language,
        "checkpoint": str(args.checkpoint),
        "episodes": episode_results,
        "successes": successes,
        "success_rate": successes / len(episode_results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"success_rate={result['success_rate']:.3f}; saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
