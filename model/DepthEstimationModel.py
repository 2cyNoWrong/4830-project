# model/DepthEstimationModel.py

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler
from model.UNetWithSemantic import UNetWithSemantic
from model.DepthDecoder import DepthDecoder

class DepthEstimationModel(nn.Module):
    """
    整体流程：
      1) VAE 编码 RGB → latent
      2) 给 latent 加噪，并用 UNetWithSemantic 预测去噪 latent
      3) 用 task‐specified decoder 将去噪 latent 解码成高分辨率深度图
    """
    def __init__(self, pretrained_sd: str, freeze_vae: bool = True):
        super().__init__()
        # 1) VAE 编码器（可选冻结）
        self.vae = AutoencoderKL.from_pretrained(pretrained_sd, subfolder="vae")
        if freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad = False
            self.vae.eval()
        self.scaling_factor = self.vae.config.scaling_factor

        # 2) UNetWithSemantic（从头初始化），输出 noise_pred == 去噪 latent
        self.unet = UNetWithSemantic()

        # 3) Task‐specified decoder：接在 UNet 输出之后
        #    UNetWithSemantic 输出的 channels == out_channels 参数，在这里为 4
        self.depth_decoder = DepthDecoder(
            encoder_dims=[self.unet.unet.config.out_channels],
            mid_channels=256,
            out_channels=1
        )

        # 4) 噪声调度器
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_sd, subfolder="scheduler")

    def encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = self.vae.encode(imgs).latent_dist.sample()
        return latents * self.scaling_factor

    def add_noise(self, latents, timesteps=None):
        noise = torch.randn_like(latents)
        if timesteps is None:
            timesteps = torch.randint(
                0, self.scheduler.num_train_timesteps,
                (latents.shape[0],), device=latents.device
            )
        noisy = self.scheduler.add_noise(latents, noise, timesteps)
        return noisy, noise, timesteps

    def forward(self, imgs: torch.Tensor, timesteps: torch.LongTensor = None):
        # A) 将 RGB 编码到 latent
        latents = self.encode_images(imgs)
        # B) 加噪与取 timesteps
        noisy_latents, noise, timesteps = self.add_noise(latents, timesteps)
        # C) UNet 去噪，得到去噪 latent（noise_pred）
        noise_pred = self.unet(noisy_latents, timesteps, imgs)
        # D) Task‐specified decoder：接去噪 latent 解码 depth
        #    depth_map 形状 [B,1,H_rgb,W_rgb]
        depth_map = self.depth_decoder(
            [noise_pred],
            target_size=imgs.shape[-2:]
        )
        return depth_map, noise, timesteps
