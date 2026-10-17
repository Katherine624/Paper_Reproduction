#一条演示轨迹中的某个时刻 → 取当前观测和未来 16 步动作 → 编成 9 个数据槽位 → 给待预测槽位加噪声 → Transformer 还原干净数据 → 计算误差并更新参数。

import argparse
import io
import json
import math
import os
import pickle
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

#DATA_ROOT、ROLLOUT_DIR、STATS_PATH、TEXT_PATH 分别指向数据集、逐回合轨迹、数值统计量和文本向量。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets" / "LIBERO-Cosmos-Policy"
ROLLOUT_DIR = DATA_ROOT / "all_episodes"
STATS_PATH = DATA_ROOT / "success_only" / "dataset_statistics.json"
TEXT_PATH = DATA_ROOT / "success_only" / "t5_embeddings.pkl"

CHUNK_SIZE = 16
IMAGE_SIZE = 64
ACTION_DIM = 7
PROPRIO_DIM = 9
GAMMA = 0.99
NUM_FRAMES = 9
LATENT_CHANNELS = 16
LATENT_HW = 8
FRAME_DIM = LATENT_CHANNELS * LATENT_HW * LATENT_HW
TEXT_DIM = 1024
HIDDEN_DIM = 256
SIGMA_MIN = 0.02
SIGMA_MAX = 3.0

#解码一张图片
def decode_image(jpeg_data):
    with Image.open(io.BytesIO(jpeg_data.tobytes())) as image:
        image = image.convert("RGB")
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )
        array = np.array(image, dtype=np.float32, copy=True)

    return torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1.0

#归一化动作或机器人状态
def normalize(values, stats, name):
    minimum = np.asarray(stats[f"{name}_min"], dtype=np.float32)
    maximum = np.asarray(stats[f"{name}_max"], dtype=np.float32)
    span = np.maximum(maximum - minimum, 1e-6)
    return (2.0 * (values - minimum) / span - 1.0).astype(np.float32)

#建立样本目录
class MiniLiberoDataset(Dataset):
    def __init__(self, only_success: bool):
        self.episodes = []

        paths = sorted(ROLLOUT_DIR.glob(
            "episode_data--suite=libero_spatial--*--task=0--*.hdf5"
        ))

        for path in paths:
            with h5py.File(path, "r") as file:
                success = bool(file.attrs["success"])
                if only_success and not success:
                    continue

                length = len(file["actions"])
                instruction = str(file.attrs["task_description"])

            self.episodes.append((path, length, success, instruction))

        if not self.episodes:
            raise FileNotFoundError("没有找到 LIBERO-Spatial task 0 的轨迹")

        self.indices = [
            (episode_id, t)
            for episode_id, (_, length, _, _) in enumerate(self.episodes)
            for t in range(length)
        ]

        with STATS_PATH.open("r", encoding="utf-8") as file:
            self.stats = json.load(file)

        with TEXT_PATH.open("rb") as file:
            cached_text = pickle.load(file)

        self.text_embeddings = {
            instruction: cached_text[instruction].squeeze(0).float().mean(dim=0)
            for _, _, _, instruction in self.episodes
        }

    def __len__(self):
        return len(self.indices)
#真正取出一条样本
    def __getitem__(self, index):
        #5.1 找到当前时刻和未来时刻
        episode_id, t = self.indices[index]
        path, length, success, instruction = self.episodes[episode_id]
        future_t = min(t + CHUNK_SIZE, length - 1)
#5.2 从 HDF5 读取观测和动作
        with h5py.File(path, "r") as file:
            wrist_t = decode_image(file["wrist_images_jpeg"][t])
            primary_t = decode_image(file["primary_images_jpeg"][t])
            wrist_future = decode_image(file["wrist_images_jpeg"][future_t])
            primary_future = decode_image(file["primary_images_jpeg"][future_t])

            proprio_t = file["proprio"][t].astype(np.float32)
            proprio_future = file["proprio"][future_t].astype(np.float32)

            end = min(t + CHUNK_SIZE, length)
            actions = file["actions"][t:end].astype(np.float32)

        if len(actions) < CHUNK_SIZE:
            padding = np.repeat(
                actions[-1:],
                CHUNK_SIZE - len(actions),
                axis=0,
            )
            actions = np.concatenate((actions, padding), axis=0)

        proprio_t = normalize(proprio_t, self.stats, "proprio")
        proprio_future = normalize(proprio_future, self.stats, "proprio")
        actions = normalize(actions, self.stats, "actions")

        discounted_return = float(success) * GAMMA ** (length - 1 - future_t)
        return_future = 2.0 * discounted_return - 1.0

        return {
            "images_t": torch.stack((wrist_t, primary_t)),
            "images_future": torch.stack((wrist_future, primary_future)),
            "proprio_t": torch.from_numpy(proprio_t),
            "proprio_future": torch.from_numpy(proprio_future),
            "actions": torch.from_numpy(actions),
            "return_future": torch.tensor(return_future, dtype=torch.float32),
            "text": self.text_embeddings[instruction],
        }

#把图片改成统一槽位尺寸
def image_to_frame(images):
    """Fixed, small replacement for the paper's video VAE tokenizer."""
    pooled = F.adaptive_avg_pool2d(images, (LATENT_HW, LATENT_HW))
    copies = (LATENT_CHANNELS + 2) // 3
    return pooled.repeat(1, copies, 1, 1)[:, :LATENT_CHANNELS]

#把数值也改成统一槽位尺寸
def vector_to_frame(values):
    """Inject a numeric vector into one image-shaped latent frame."""
    flat = values.reshape(values.shape[0], -1)
    copies = (FRAME_DIM + flat.shape[1] - 1) // flat.shape[1]
    tiled = flat.repeat(1, copies)[:, :FRAME_DIM]
    return tiled.reshape(-1, LATENT_CHANNELS, LATENT_HW, LATENT_HW)

#把一批样本排成 9 个槽位
def build_latent_sequence(batch):
    """Order LIBERO data as (blank, current state, action, future state, value)."""
    batch_size = batch["actions"].shape[0]
    blank = batch["actions"].new_zeros(
        batch_size, LATENT_CHANNELS, LATENT_HW, LATENT_HW
    )

    frames = [
        blank,
        vector_to_frame(batch["proprio_t"]),
        image_to_frame(batch["images_t"][:, 0]),
        image_to_frame(batch["images_t"][:, 1]),
        vector_to_frame(batch["actions"]),
        vector_to_frame(batch["proprio_future"]),
        image_to_frame(batch["images_future"][:, 0]),
        image_to_frame(batch["images_future"][:, 1]),
        vector_to_frame(batch["return_future"].unsqueeze(1)),
    ]
    return torch.stack(frames, dim=1).flatten(start_dim=2)

#规定哪些已知、哪些要预测
def condition_mask(mode, batch_size, device):
    """True marks clean frames supplied as conditions to the denoiser."""
    prefix_lengths = {"policy": 4, "world": 5, "value": 8}
    if mode not in prefix_lengths:
        raise ValueError(f"Unknown training mode: {mode}")
    frame_ids = torch.arange(NUM_FRAMES, device=device)
    return (frame_ids < prefix_lengths[mode]).unsqueeze(0).expand(batch_size, -1)


"""
Cosmos Policy：带噪数据 → Transformer → 预测干净数据 x₀
π0：          噪声与动作的中间状态 → Transformer → 预测移动方向/速度
层次	    回答的问题	                  例子
网络架构	用什么结构处理输入、计算输出？	Transformer、U-Net、MLP
训练目标	希望网络预测什么？预测错了怎样扣分？	去噪预测干净数据、流匹配预测速度
生成过程	训练好后怎样一步步产生动作？	逐步去噪、沿预测的速度积分
"""
#神经网络
class MiniDenoiser(nn.Module):
    """Small frame-token Transformer replacing the 2B diffusion backbone."""

    def __init__(self):
        super().__init__()
        self.frame_projection = nn.Linear(FRAME_DIM, HIDDEN_DIM)
        self.position = nn.Parameter(torch.zeros(1, NUM_FRAMES, HIDDEN_DIM))
        self.condition_embedding = nn.Embedding(2, HIDDEN_DIM)
        self.text_projection = nn.Sequential(
            nn.LayerNorm(TEXT_DIM),
            nn.Linear(TEXT_DIM, HIDDEN_DIM),
        )
        self.noise_projection = nn.Sequential(
            nn.Linear(1, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=8,
            dim_feedforward=4 * HIDDEN_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.output_projection = nn.Linear(HIDDEN_DIM, FRAME_DIM)
#一次前向传播
    def forward(self, noisy_frames, sigma, text, known_frames):
        frame_features = self.frame_projection(noisy_frames)
        text_features = self.text_projection(text).unsqueeze(1)
        noise_features = self.noise_projection(sigma.log().unsqueeze(1)).unsqueeze(1)
        mask_features = self.condition_embedding(known_frames.long())
        hidden = (
            frame_features
            + self.position
            + text_features
            + noise_features
            + mask_features
        )
        return self.output_projection(self.transformer(hidden))

#一次损失如何产生
def diffusion_loss(model, batch, mode):
    """Predict clean target frames after corrupting them with Gaussian noise."""
#生成标准答案和掩码
    clean_frames = build_latent_sequence(batch)
    batch_size = clean_frames.shape[0]
    known_frames = condition_mask(mode, batch_size, clean_frames.device)
#每个样本随机选噪声强度
    log_sigma = torch.empty(batch_size, device=clean_frames.device).uniform_(
        math.log(SIGMA_MIN), math.log(SIGMA_MAX)
    )
    sigma = log_sigma.exp()
#给数据加高斯噪声
    noisy_frames = clean_frames + sigma[:, None, None] * torch.randn_like(clean_frames)
#把已知条件换回干净值
    noisy_frames = torch.where(known_frames[:, :, None], clean_frames, noisy_frames)
#模型预测并只计算待预测位置的误差
    predicted_frames = model(noisy_frames, sigma, batch["text"], known_frames)
    target_mask = (~known_frames).unsqueeze(-1).to(clean_frames.dtype)
    squared_error = (predicted_frames - clean_frames).square() * target_mask
    return squared_error.sum() / (target_mask.sum() * FRAME_DIM)
#读取运行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Train a small Cosmos Policy-style LIBERO model")
    parser.add_argument("--steps", type=int, default=1000, help="Total optimizer updates")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "libero_mini")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    for name in ("steps", "batch_size", "accum_steps", "log_every", "val_every", "val_batches", "save_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.workers < 0 or args.lr <= 0:
        parser.error("--workers must be nonnegative and --lr must be positive")
    return args

#建立一个加载器
def make_loader(dataset, selected_paths, batch_size, workers, shuffle):
    sample_ids = [
        sample_id
        for sample_id, (episode_id, _) in enumerate(dataset.indices)
        if dataset.episodes[episode_id][0] in selected_paths
    ]
    if not sample_ids:
        raise ValueError("The selected LIBERO split contains no samples")
    return DataLoader(
        Subset(dataset, sample_ids),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )

#建立训练集和验证集
def make_loaders(batch_size, workers):
    policy_dataset = MiniLiberoDataset(only_success=True)
    rollout_dataset = MiniLiberoDataset(only_success=False)
    all_paths = sorted(episode[0] for episode in rollout_dataset.episodes)
    validation_paths = set(all_paths[::10])
    training_paths = set(all_paths) - validation_paths

    train_loaders = {
        "policy": make_loader(policy_dataset, training_paths, batch_size, workers, True),
        "rollout": make_loader(rollout_dataset, training_paths, batch_size, workers, True),
    }
    validation_loaders = {
        "policy": make_loader(policy_dataset, validation_paths, batch_size, 0, False),
        "rollout": make_loader(rollout_dataset, validation_paths, batch_size, 0, False),
    }
    print(
        f"Train episodes: {len(training_paths)}; validation episodes: {len(validation_paths)}; "
        f"train samples: policy={len(train_loaders['policy'].dataset)}, "
        f"rollout={len(train_loaders['rollout'].dataset)}",
        flush=True,
    )
    task_description = rollout_dataset.episodes[0][3]
    return train_loaders, validation_loaders, rollout_dataset.stats, task_description

#把数据搬到计算设备
def move_batch(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}

#持续取下一批
def next_training_batch(mode, loaders, iterators):
    loader_name = "policy" if mode == "policy" else "rollout"
    if loader_name not in iterators:
        iterators[loader_name] = iter(loaders[loader_name])
    try:
        return next(iterators[loader_name])
    except StopIteration:
        iterators[loader_name] = iter(loaders[loader_name])
        return next(iterators[loader_name])

#数据验证
@torch.no_grad()
def evaluate(model, loaders, device, max_batches):
    model.eval()
    results = {}
    cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(12345)
        for mode in ("policy", "world", "value"):
            loader_name = "policy" if mode == "policy" else "rollout"
            total_loss = 0.0
            total_samples = 0
            for batch_number, batch in enumerate(loaders[loader_name]):
                if batch_number >= max_batches:
                    break
                batch = move_batch(batch, device)
                batch_size = batch["actions"].shape[0]
                total_loss += diffusion_loss(model, batch, mode).item() * batch_size
                total_samples += batch_size
            results[mode] = total_loss / total_samples
    model.train()
    return results

#保存可以继续训练的状态
def save_checkpoint(path, model, optimizer, scheduler, step, args, stats, task_description, validation, best_policy_val):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "config": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "dataset_stats": stats,
        "task_description": task_description,
        "validation": validation,
        "best_policy_val": best_policy_val,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)

#把前面所有函数串成训练
"""
创建数据加载器和模型
    ↓
选一种训练模式，取一批轨迹样本
    ↓
把当前观测、动作、未来状态和回报排成 9 个槽位
    ↓
确定哪些槽位已知，给待预测槽位加噪
    ↓
Transformer 预测干净的 9 个槽位
    ↓
只对待预测槽位计算误差
    ↓
反向传播 → 更新模型参数
    ↓
定期验证、保存权重 → 继续下一步
"""
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loaders, validation_loaders, stats, task_description = make_loaders(
        args.batch_size, args.workers
    )
    model = MiniDenoiser().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1
    )

    start_step = 0
    best_policy_val = float("inf")
    last_validation = {}
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
        best_policy_val = float(checkpoint.get("best_policy_val", float("inf")))
        last_validation = checkpoint.get("validation", {})
        print(f"Resumed from step {start_step}: {args.resume}", flush=True)

    print(
        f"Device: {device}; parameters: {sum(parameter.numel() for parameter in model.parameters()):,}; "
        f"optimizer steps: {start_step + 1}..{args.steps}",
        flush=True,
    )
    iterators = {}
    recent_losses = []
    recent_modes = {"policy": 0, "world": 0, "value": 0}
    started_at = time.perf_counter()

    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        microbatch_losses = []
        for _ in range(args.accum_steps):
            mode = random.choices(("policy", "world", "value"), weights=(2, 1, 1))[0]
            batch = move_batch(next_training_batch(mode, train_loaders, iterators), device)
            loss = diffusion_loss(model, batch, mode)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {mode} loss at step {step}: {loss.item()}")
            (loss / args.accum_steps).backward()
            microbatch_losses.append(loss.item())
            recent_modes[mode] += 1

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        recent_losses.append(sum(microbatch_losses) / len(microbatch_losses))

        if step == 1 or step % args.log_every == 0:
            elapsed = time.perf_counter() - started_at
            print(
                f"step={step}/{args.steps} loss={sum(recent_losses) / len(recent_losses):.5f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} modes={recent_modes} elapsed={elapsed:.0f}s",
                flush=True,
            )
            recent_losses.clear()
            recent_modes = {"policy": 0, "world": 0, "value": 0}

        if step % args.val_every == 0 or step == args.steps:
            last_validation = evaluate(model, validation_loaders, device, args.val_batches)
            print(f"validation step={step}: {last_validation}", flush=True)
            if last_validation["policy"] < best_policy_val:
                best_policy_val = last_validation["policy"]
                save_checkpoint(
                    args.output_dir / "best.pt", model, optimizer, scheduler,
                    step, args, stats, task_description, last_validation, best_policy_val,
                )

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(
                args.output_dir / "last.pt", model, optimizer, scheduler,
                step, args, stats, task_description, last_validation, best_policy_val,
            )

    print(f"Training complete. Checkpoints: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
