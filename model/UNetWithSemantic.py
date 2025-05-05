import torch
from torch import nn
from diffusers import UNet2DConditionModel

from model.Seecoder import SeeCoder


class UNetWithSemantic(nn.Module):
    """
    UNet 模型，集成 SeeCoder 提取的语义向量，
    直接调用 UNet2DConditionModel.forward，自动处理时间嵌入和跨注意力。
    """
    def __init__(self):
        super().__init__()
        # 从头初始化 UNet2DConditionModel
        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            layers_per_block=[2, 2, 2, 2],
            block_out_channels=[320, 640, 1280, 1280],
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            attention_head_dim=[64, 64, 64, 64],
            cross_attention_dim=768,
            norm_num_groups=4,
        )
        self.seecoder = SeeCoder()

    def forward(self,
                noisy_latents: torch.FloatTensor,
                timesteps: torch.LongTensor,
                encoder_image: torch.FloatTensor
    ) -> torch.FloatTensor:
        # 提取 SeeCoder 语义特征
        sem_feats = self.seecoder(encoder_image)
        # 直接调用 UNet2DConditionModel.forward 自动处理 conv_in, timesteps embedding, down/up blocks, conv_out
        out = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=sem_feats
        )
        # 支持不同返回格式
        if isinstance(out, tuple):
            noise_pred = out[0]
        else:
            noise_pred = out.sample
        return noise_pred