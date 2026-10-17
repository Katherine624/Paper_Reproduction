"""Train the action-conditioned τ0-WM Simulator on real taco_play data."""

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.utils.checkpoint
import yaml
from diffusers.optimization import get_scheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = PROJECT_ROOT / "upstream" / "tau-0-wm"
sys.path.insert(0, str(UPSTREAM_ROOT))

from data.example_dataset_sim import CustomLeRobotDataset
from models.wan_2_2_models.scheduler.fm_solvers_unipc import (
    FlowUniPCMultistepScheduler,
)
from models.wan_2_2_models.text_encoder.t5 import T5EncoderModel
from models.wan_2_2_models.transformers.model_sim import WanModel
from models.wan_2_2_models.vae.vae2_2 import Wan2_2_VAE_Batch
from utils.data_utils import apply_color_jitter_to_video
from utils.model_utils import load_diffusion_model


N_MEMORY_FRAMES = 3


def load_config() -> dict:
    """Read the released Simulator settings and bind local checkpoint paths."""
    config_path = (
        UPSTREAM_ROOT / "configs" / "tau_model" / "posttrain_sim_taco_play_abs.yaml"
    )
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["diffusion_model"]["model_path"] = str(
        PROJECT_ROOT / "weights" / "simulator" / "diffusion_pytorch_model.bin"
    )
    config["vae_path"] = str(
        PROJECT_ROOT / "weights" / "wan2.2-ti2v-5b" / "Wan2.2_VAE.pth"
    )
    config["text_encoder"]["checkpoint_path"] = str(
        PROJECT_ROOT
        / "weights"
        / "wan2.2-ti2v-5b"
        / "models_t5_umt5-xxl-enc-bf16.pth"
    )
    config["text_encoder"]["tokenizer_path"] = str(
        PROJECT_ROOT / "weights" / "wan2.2-ti2v-5b" / "google" / "umt5-xxl"
    )
    config["output_dir"] = str(
        PROJECT_ROOT / "runs" / "checkpoints" / "sim-taco-play-subset100"
    )
    config["batch_size"] = 1
    config["train_epochs"] = 1
    config["train_steps"] = 100
    config["steps_to_save"] = 100
    config["steps_to_log"] = 10
    config["lr_warmup_steps"] = 10
    config["subset_size"] = 100
    config["gradient_checkpointing"] = True
    return config


def build_sim_dataloader(config: dict) -> DataLoader:
    """The released dataset produces video, memory frames, actions and rewards."""
    data_config_path = PROJECT_ROOT / "workspace" / "configs" / "taco_abs_joint.yaml"
    with data_config_path.open(encoding="utf-8") as file:
        data_config = yaml.safe_load(file)

    data_config["data"]["data_roots"] = [str(PROJECT_ROOT / "data/raw/taco_play")]
    data_config["data"]["statistic_files"] = [str(
        UPSTREAM_ROOT / "configs/data/stat_file/taco_play_abs.json"
    )]
    dataset = CustomLeRobotDataset(
        **data_config["data"],
        add_mem_cond=True,
        n_mem=N_MEMORY_FRAMES,
        return_reward=True,
    )
    generator = torch.Generator().manual_seed(config["seed"])
    indices = torch.randperm(len(dataset), generator=generator)[
        :config["subset_size"]
    ].tolist()
    dataset = Subset(dataset, indices)
    print(f"Simulator subset: {len(dataset)} samples", flush=True)
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["dataloader_num_workers"],
        pin_memory=config["pin_memory"],
        shuffle=True,
        drop_last=True,
        generator=generator,
    )


def load_components(config: dict, device: torch.device):
    """Load frozen VAE/T5 and initialize Simulator from its released weights."""
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }.get(config["mixed_precision"], torch.float32)

    vae = Wan2_2_VAE_Batch(
        vae_pth=config["vae_path"],
        device=device,
        dtype=dtype,
    )
    vae.vae_encoding_mode = config["vae_encoding_mode"]
    vae.model.eval().requires_grad_(False)

    text_encoder = T5EncoderModel(
        device=device,
        dtype=dtype,
        **config["text_encoder"],
    )
    text_encoder.model.eval().requires_grad_(False)

    model = load_diffusion_model(
        model_cls=WanModel,
        model_dir=config["diffusion_model"]["model_path"],
        load_weights=config["load_weights"],
        **config["diffusion_model"]["config"],
    ).to(device=device, dtype=dtype)
    model.gradient_checkpointing = config["gradient_checkpointing"]
    return model, vae, text_encoder


def build_optimizer(model: WanModel, config: dict) -> torch.optim.Optimizer:
    """Use the same trainable-parameter groups as the released Simulator trainer."""
    train_modes = set(config["train_mode"].lower().split("+"))
    trainable_params = []

    for name, parameter in model.named_parameters():
        if "action_" in name:
            trainable = "action" in train_modes
        elif "vlm_interface" in name:
            trainable = "vlm" in train_modes
        elif "reward" in name:
            trainable = "reward" in train_modes
        else:
            trainable = "video" in train_modes

        parameter.requires_grad_(trainable)
        if trainable:
            trainable_params.append(parameter)

    return torch.optim.AdamW(
        trainable_params,
        lr=float(config["lr"]),
        betas=(float(config["beta1"]), float(config["beta2"])),
        eps=float(config["epsilon"]),
        weight_decay=float(config["weight_decay"]),
    )


def encode_sim_video(
    video: torch.Tensor,
    memory: torch.Tensor,
    vae: Wan2_2_VAE_Batch,
    use_color_jitter: bool,
) -> tuple[torch.Tensor, int]:
    """Encode memory images separately, then prepend them to the future clip."""
    batch_size, _, n_views, _, _, _ = video.shape
    n_mem = memory.shape[3]
    video = rearrange(video, "b c v t h w -> (b v) c t h w")
    memory = rearrange(memory, "b c v t h w -> (b v) c t h w")

    if use_color_jitter:
        combined = apply_color_jitter_to_video(torch.cat((memory, video), dim=2))
        memory, video = torch.split(
            combined, [n_mem, video.shape[2]], dim=2
        )

    memory_as_single_frames = rearrange(
        memory, "bv c (t f) h w -> (bv t) c f h w", t=n_mem
    )
    with torch.no_grad():
        future_latents = torch.stack(vae.encode(video, chunk=3), dim=0)
        memory_latents = torch.stack(
            vae.encode(memory_as_single_frames, chunk=3), dim=0
        )

    future_latents = rearrange(
        future_latents,
        "(b v) c f h w -> b v c f h w",
        b=batch_size,
        v=n_views,
    )
    memory_latents = rearrange(
        memory_latents,
        "(b v t) c f h w -> b v c (t f) h w",
        b=batch_size,
        v=n_views,
        t=n_mem,
    )
    latents = torch.cat((memory_latents, future_latents), dim=3)
    latents = rearrange(latents, "b v c f h w -> b c f h (v w)")
    return latents, n_mem

#给需要预测的未来部分加噪；同时给奖励加噪。
def compute_sim_loss(
    batch: dict,
    model: WanModel,
    vae: Wan2_2_VAE_Batch,
    text_encoder: T5EncoderModel,
    video_sigmas: torch.Tensor,
    reward_sigmas: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flow-matching losses for future video and reward, conditioned on actions."""
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype

    video = batch["video"].to(device=device, dtype=dtype)
    memory = batch["mem"].to(device=device, dtype=dtype)
    actions = batch["actions"].to(device=device, dtype=dtype)
    rewards = batch["reward"].to(device=device, dtype=dtype)

    latents, n_mem = encode_sim_video(
        video, memory, vae, config["use_color_jitter"]
    )

    # Memory latents and the first future latent are known observations.
    future_mask = torch.ones_like(latents)
    future_mask[:, :, : n_mem + 1] = 0
    video_noise = (
        torch.randn_like(latents) * future_mask
        + latents * (1 - future_mask)
    )
    video_indices = torch.randint(
        0, 1000, (latents.shape[0],), device=device
    )
    video_sigma = video_sigmas[video_indices].to(latents.dtype)
    video_sigma = video_sigma[:, None, None, None, None]
    noisy_latents = (1 - video_sigma) * latents + video_sigma * video_noise
    video_target = video_noise - latents

    patch_t, patch_h, patch_w = model.patch_size
    seq_len = (
        latents.shape[2] * latents.shape[3] * latents.shape[4]
        // (patch_h * patch_w)
    )
    video_timestep = (
        future_mask[:, :1, ::patch_t, ::patch_h, ::patch_w]
        * (video_sigma * 1000)
    ).flatten(1)

    reward_noise = torch.randn_like(rewards)
    reward_indices = torch.randint(
        0, 1000, (rewards.shape[0],), device=device
    )
    reward_sigma = reward_sigmas[reward_indices].to(rewards.dtype)
    reward_sigma = reward_sigma[:, None, None]
    noisy_rewards = (1 - reward_sigma) * rewards + reward_sigma * reward_noise
    reward_target = reward_noise - rewards
    reward_timestep = (reward_sigma * 1000).expand(
        -1, rewards.shape[1], -1
    )

    captions = [
        "" if random.random() < config["caption_dropout_p"] else caption
        for caption in batch["caption"]
    ]
    action_drop = (
        torch.rand((actions.shape[0], 1, 1), device=device)
        < config["action_dropout_p"]
    )
    actions = actions.masked_fill(action_drop, 0)

    with torch.no_grad():
        context = text_encoder(captions, device)
    with torch.autocast(
        device_type="cuda",
        dtype=dtype if dtype != torch.float32 else torch.bfloat16,
        enabled=dtype != torch.float32,
    ):
        prediction = model(
            list(noisy_latents.unbind(dim=0)),
            t=video_timestep,
            context=context,
            seq_len=seq_len,
            action_states=actions,
            reward_states=noisy_rewards,
            reward_timestep=reward_timestep,
            return_video=True,
            return_reward=True,
        )

    predicted_video = torch.stack(prediction["video"], dim=0)
    predicted_reward = prediction["reward"]
    loss_video = (
        (predicted_video.float() - video_target.float()).square()
        * future_mask.float()
    ).mean()
    loss_reward = (
        predicted_reward.float() - reward_target.float()
    ).square().mean()
    total_loss = (
        loss_video * float(config.get("video_loss_scale", 1.0))
        + loss_reward * float(config.get("action_loss_scale", 1.0))
    )
    return total_loss, loss_video, loss_reward

#把当前训练后的 Simulator 权重保存到 step_步数 文件夹。它目前只保存模型权重，不保存优化器状态。
def save_sim_checkpoint(model: WanModel, output_dir: str, step: int) -> None:
    save_dir = Path(output_dir) / f"step_{step}"
    model.save_pretrained(save_dir, safe_serialization=False)


def train_sim(
    dataloader: DataLoader,
    model: WanModel,
    vae: Wan2_2_VAE_Batch,
    text_encoder: T5EncoderModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    video_sigmas: torch.Tensor,
    reward_sigmas: torch.Tensor,
    config: dict,
) -> None:
    global_step = 0
    last_saved_step = 0

    for epoch in range(config["train_epochs"]):
        model.train()
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            loss, video_loss, reward_loss = compute_sim_loss(
                batch, model, vae, text_encoder,
                video_sigmas, reward_sigmas, config,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(config["max_grad_norm"])
            )
            optimizer.step()
            lr_scheduler.step()
            global_step += 1

            if global_step % config["steps_to_log"] == 0:
                print(
                    f"epoch={epoch + 1} step={global_step} "
                    f"loss={loss.item():.4f} "
                    f"video={video_loss.item():.4f} "
                    f"reward={reward_loss.item():.4f} "
                    f"lr={lr_scheduler.get_last_lr()[0]:.8f}"
                )

            if (
                global_step % config["steps_to_save"] == 0
                or global_step >= config["train_steps"]
            ):
                save_sim_checkpoint(model, config["output_dir"], global_step)
                last_saved_step = global_step

            if global_step >= config["train_steps"]:
                return

    if global_step > 0 and global_step != last_saved_step:
        save_sim_checkpoint(model, config["output_dir"], global_step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Simulator on taco_play")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--subset-size", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = load_config()
    if args.init_checkpoint is not None:
        config["diffusion_model"]["model_path"] = str(args.init_checkpoint)
    if args.train_steps is not None:
        config["train_steps"] = args.train_steps
    if args.subset_size is not None:
        config["subset_size"] = args.subset_size
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    config["steps_to_save"] = config["train_steps"]
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device("cuda")

    dataloader = build_sim_dataloader(config)
    model, vae, text_encoder = load_components(config, device)
    optimizer = build_optimizer(model, config)
    lr_scheduler = get_scheduler(
        name=config["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["lr_warmup_steps"],
        num_training_steps=config["train_steps"],
    )
    video_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=5.0,
        use_dynamic_shifting=False,
    )
    reward_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )

    dtype = next(model.parameters()).dtype
    video_sigmas = video_scheduler.sigmas.to(device=device, dtype=dtype)
    reward_sigmas = reward_scheduler.sigmas.to(device=device, dtype=dtype)
    train_sim(
        dataloader, model, vae, text_encoder, optimizer, lr_scheduler,
        video_sigmas, reward_sigmas, config,
    )


if __name__ == "__main__":
    main()
