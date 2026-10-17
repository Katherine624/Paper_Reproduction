import argparse
import torch
import torch.utils.checkpoint
from einops import rearrange
import sys
from pathlib import Path

import yaml
from torch.utils.data import DataLoader, Subset
from diffusers.optimization import get_scheduler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "upstream" / "tau-0-wm"))

from data.example_dataset import CustomLeRobotDataset
from models.wan_2_2_models.transformers.model import WanModel
from models.wan_2_2_models.vae.vae2_2 import Wan2_2_VAE_Batch
from models.wan_2_2_models.text_encoder.t5 import T5EncoderModel
from utils.model_utils import load_diffusion_model

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from models.wan_2_2_models.scheduler.fm_solvers_unipc import (
    FlowUniPCMultistepScheduler,
)
#动作加噪，制作模型的训练题目
def prepare_action_training(
    actions: torch.Tensor,
    scheduler_sigmas: torch.Tensor,
):
    """为一批真实动作构造 Flow Matching 训练输入和目标。"""
    batch_size = actions.shape[0]

    step_indices = torch.randint(
        0, 1000, (batch_size,), device=actions.device
    )
    sigma = scheduler_sigmas[step_indices].to(actions.dtype)
    sigma = sigma[:, None, None]

    noise = torch.randn_like(actions)
    noisy_actions = (1 - sigma) * actions + sigma * noise
    target = noise - actions

    action_timestep = (sigma * 1000).expand(
        -1, actions.shape[1], -1
    )
    return noisy_actions, target, action_timestep

#视频加噪
def prepare_video_training(
    latents: torch.Tensor,
    scheduler_sigmas: torch.Tensor,
):
    """构造视频潜变量的 Flow Matching 训练输入和目标。"""
    batch_size = latents.shape[0]

    # 第一帧是已知观测；只有未来帧需要预测
    future_mask = torch.ones_like(latents)
    future_mask[:, :, 0] = 0.0

    noise = (
        torch.randn_like(latents) * future_mask
        + latents * (1 - future_mask)
    )

    step_indices = torch.randint(
        0, 1000, (batch_size,), device=latents.device
    )
    sigma = scheduler_sigmas[step_indices].to(latents.dtype)
    sigma = sigma[:, None, None, None, None]

    noisy_latents = (1 - sigma) * latents + sigma * noise
    target = noise - latents

    return noisy_latents, target, future_mask, sigma


#把 taco_play 的真实多视角视频交给 VAE 压缩，得到后续训练 VAM 使用的 latents。
def encode_video_latents(video: torch.Tensor, vae) -> torch.Tensor:
    """用冻结的 VAE 将多视角真实视频压缩为潜变量。"""
    batch_size, _, n_views, _, _, _ = video.shape

    # 将视角合并到批量维度，逐个视角进行编码
    video = rearrange(video, "b c v t h w -> (b v) c t h w")

    with torch.no_grad():
        latents = vae.encode(video, chunk=3)

    latents = torch.stack(latents, dim=0)

    # 将多个视角的特征沿宽度方向拼接
    latents = rearrange(
        latents,
        "(b v) c f h w -> b c f h (v w)",
        b=batch_size,
        v=n_views,
    )
    return latents






#读取一批真实数据 → VAE编码 → 视频和动作分别加噪 → VAM预测 → 计算联合损失
def compute_vam_loss(
    batch,
    vae,
    text_encoder,
    model,
    video_sigmas,
    action_sigmas,
):
    """完成一批数据的 VAM 前向计算，并返回联合损失。"""
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype

    video = batch["video"].to(device=device, dtype=dtype)
    actions = batch["actions"].to(device=device, dtype=dtype)
    state = batch["state"].to(device=device, dtype=dtype)
    captions = batch["caption"]

    latents = encode_video_latents(video, vae)

    noisy_latents, video_target, future_mask, sigma = (
        prepare_video_training(
            latents,
            video_sigmas.to(device),
        )
    )
    noisy_actions, action_target, action_timestep = (
        prepare_action_training(
            actions,
            action_sigmas.to(device),
        )
    )

    with torch.no_grad():
        context = text_encoder(captions, device)

    patch_t, patch_h, patch_w = model.patch_size
    seq_len = (
        latents.shape[2] * latents.shape[3] * latents.shape[4]
        // (patch_h * patch_w)
    )
    video_timestep = (
        future_mask[:, :1, ::patch_t, ::patch_h, ::patch_w]
        * (sigma * 1000)
    ).flatten(1)

    with torch.autocast(device_type="cuda", dtype=dtype):
        prediction = model(
            list(noisy_latents.unbind(dim=0)),
            t=video_timestep,
            context=context,
            seq_len=seq_len,
            action_states=noisy_actions,
            action_timestep=action_timestep,
            history_action_state=state,
            return_video=True,
            return_action=True,
        )

    predicted_video = torch.stack(prediction["video"], dim=0)
    predicted_action = prediction["action"]

    loss_video = (
        (predicted_video - video_target).square() * future_mask
    ).mean()
    loss_action = (
        predicted_action - action_target
    ).square().mean()

    total_loss = loss_video + loss_action
    return total_loss, loss_video, loss_action




def build_vam_optimizer(model, vae, text_encoder):
    # VAE 和文字编码器只负责提供输入，不参与这次训练
    vae.model.eval().requires_grad_(False)
    text_encoder.model.eval().requires_grad_(False)

    trainable_params = []
#将训练的权重参数加入到trainable中
    for name, param in model.named_parameters():
        if "action_" in name:
            trainable = True
        elif "vlm_interface" in name:
            trainable = False
        else:
            trainable = True

        param.requires_grad_(trainable)

        if trainable:
            trainable_params.append(param)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=5e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-5,
    )
    return optimizer



#用一批真实数据更新一次 VAM 权重
def train_vam_step(
    batch, model, vae, text_encoder, optimizer,
    video_sigmas, action_sigmas,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss, loss_video, loss_action = compute_vam_loss(
        batch, vae, text_encoder, model,
        video_sigmas, action_sigmas,
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), loss_video.item(), loss_action.item()
"""
model.train()：让 VAM 进入训练模式。
optimizer.zero_grad(...)：清掉上一次留下的梯度。
compute_vam_loss(...)：拿这一批视频和动作，让模型预测，并算出误差。
loss.backward()：计算每个可训练权重应如何调整；此时权重还没变。
clip_grad_norm_(...)：限制过大的梯度，避免更新不稳定。
optimizer.step()：真正修改 VAM 权重。
loss.item()：把损失转成普通数字，方便之后打印训练进度。
"""

def save_vam_checkpoint(model, output_dir, step):
    save_dir = Path(output_dir) / f"step_{step}"
    model.save_pretrained(save_dir, safe_serialization=False)


#一批数据更新一次”重复执行
def train_vam(
    dataloader, model, vae, text_encoder, optimizer, lr_scheduler,
    video_sigmas, action_sigmas, train_epochs, train_steps,
    output_dir, steps_to_save,
):
    global_step = 0

    for epoch in range(train_epochs):
        for batch in dataloader:
            loss, video_loss, action_loss = train_vam_step(
                batch, model, vae, text_encoder, optimizer,
                video_sigmas, action_sigmas,
            )
            lr_scheduler.step()
            global_step += 1

            if global_step % steps_to_save == 0 or global_step >= train_steps:
                save_vam_checkpoint(model, output_dir, global_step)

            if global_step == 1 or global_step % 10 == 0:
                print(
                    f"step={global_step} "
                    f"loss={loss:.4f} "
                    f"video={video_loss:.4f} "
                    f"action={action_loss:.4f}",
                    flush=True,
                )

            if global_step >= train_steps:
                return

    if global_step > 0 and global_step % steps_to_save != 0:
        save_vam_checkpoint(model, output_dir, global_step)




#dataloader
def build_taco_dataloader(batch_size, subset_size, seed):
    config_path = (
        PROJECT_ROOT / "workspace" / "configs" / "taco_abs_joint.yaml"
    )

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["data"]["data_roots"] = [str(PROJECT_ROOT / "data/raw/taco_play")]
    config["data"]["statistic_files"] = [str(
        PROJECT_ROOT / "upstream/tau-0-wm/configs/data/stat_file/taco_play_abs.json"
    )]
    dataset = CustomLeRobotDataset(**config["data"])
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    dataset = Subset(dataset, indices)
    print(f"VAM subset: {len(dataset)} samples", flush=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        generator=generator,
    )
    return dataloader



#加载函数：加载 VAM、VAE 和文字编码器
def load_vam_components(device, init_checkpoint=None):
    config_path = (
        PROJECT_ROOT / "workspace" / "configs" / "vam_taco_play.yaml"
    )
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if init_checkpoint is not None:
        config["diffusion_model"]["model_path"] = str(init_checkpoint)

    dtype = torch.bfloat16

    vae = Wan2_2_VAE_Batch(
        vae_pth=config["vae_path"],
        device=device,
        dtype=dtype,
    )

    text_encoder = T5EncoderModel(
        device=device,
        dtype=dtype,
        **config["text_encoder"],
    )

    model = load_diffusion_model(
        model_cls=WanModel,
        model_dir=config["diffusion_model"]["model_path"],
        load_weights=True,
        **config["diffusion_model"]["config"],
    ).to(device=device, dtype=dtype)
    model.gradient_checkpointing = config["gradient_checkpointing"]

    return model, vae, text_encoder, config




#main函数:加载模型 → 准备真实数据和加噪规则 → 逐批计算损失 → 更新 VAM 权重。
def main():
    parser = argparse.ArgumentParser(description="Train the VAM on taco_play")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--subset-size", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda")

    model, vae, text_encoder, config = load_vam_components(
        device, args.init_checkpoint
    )#加载模型
    if args.train_steps is not None:
        config["train_steps"] = args.train_steps
    if args.subset_size is not None:
        config["subset_size"] = args.subset_size
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    config["steps_to_save"] = config["train_steps"]
    torch.manual_seed(config["seed"])
    dataloader = build_taco_dataloader(
        config["batch_size"], config["subset_size"], config["seed"]
    )#创建数据读取器
    optimizer = build_vam_optimizer(model, vae, text_encoder)
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
    )#视频加噪规则
    action_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )#动作加噪规则

    train_vam(
        dataloader, model, vae, text_encoder, optimizer, lr_scheduler,
        video_scheduler.sigmas,
        action_scheduler.sigmas,
        config["train_epochs"],
        config["train_steps"],
        config["output_dir"],
        config["steps_to_save"],
    )#正式进入训练循环


if __name__ == "__main__":
    main()
