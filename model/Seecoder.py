import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.swin_transformer import SwinTransformer, SwinTransformerBlock

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True)[0]
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn

class SEBlock(nn.Module):
    def __init__(self, blk: SwinTransformerBlock):
        super().__init__()
        self.block = blk
        for p in self.block.parameters():
            p.requires_grad = False
        dim = blk.dim
        self.dilated_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        x = self.block(x)
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        feat = x.permute(0,2,1).view(B, C, H, W)
        feat = self.dilated_conv(feat)
        feat = self.spatial_attn(feat)
        return feat.flatten(2).transpose(1,2)

class SeeCoderBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        base: SwinTransformer = timm.create_model(
            "swin_large_patch4_window7_224", pretrained=True
        )
        self.patch_embed = base.patch_embed
        self.pos_drop   = base.pos_drop
        self.norm       = base.norm
        self.layers     = nn.ModuleList()
        self.downsamples= nn.ModuleList()
        for layer in base.layers:
            wrapped_blocks = nn.ModuleList([SEBlock(blk) for blk in layer.blocks])
            self.layers.append(wrapped_blocks)
            # preserve downsample (PatchMerging) or identity
            if hasattr(layer, 'downsample') and layer.downsample is not None:
                self.downsamples.append(layer.downsample)
            else:
                self.downsamples.append(nn.Identity())

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        feats = []
        # iterate through stages
        for blocks, downsample in zip(self.layers, self.downsamples):
            for blk in blocks:
                x = blk(x)
            # collect feature BEFORE downsampling
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            feats.append(x.permute(0,2,1).view(B, C, H, W))
            # then downsample for next stage
            x = downsample(x)
        return feats  # channels [192,384,768,1536]

class SeeDecoder(nn.Module):
    def __init__(self, in_dims, embed_dim=768):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(c, embed_dim, 1) for c in in_dims])
        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
        )
    def forward(self, feats):
        target = feats[0].shape[-2:]
        up = []
        for proj, f in zip(self.projs, feats):
            y = proj(f)
            if y.shape[-2:] != target:
                y = F.interpolate(y, size=target, mode='bilinear', align_corners=False)
            up.append(y)
        x = sum(up)
        return self.refine(x)

class QueryTransformer(nn.Module):
    def __init__(self, local_q=144, global_q=4, d_model=768, nhead=8, num_layers=6):
        super().__init__()
        self.local_q  = nn.Parameter(torch.randn(local_q, d_model))
        self.global_q = nn.Parameter(torch.randn(global_q, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.self_attn  = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True), num_layers
        )
    def forward(self, x):
        B, C, H, W = x.shape
        ctx = x.flatten(2).transpose(1,2)
        lq = self.local_q.unsqueeze(0).expand(B, -1, -1)
        gq = self.global_q.unsqueeze(0).expand(B, -1, -1)
        l_out, _ = self.cross_attn(lq, ctx, ctx)
        q_all = torch.cat([l_out, gq], dim=1)
        return self.self_attn(q_all)

class SeeCoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SeeCoderBackbone()
        self.decoder  = SeeDecoder([192,384,768,1536], embed_dim=768)
        self.query_net= QueryTransformer()

    def forward(self, img):
        tgt_h, tgt_w = self.backbone.patch_embed.img_size
        if img.shape[-2:] != (tgt_h, tgt_w):
            img = F.interpolate(img, size=(tgt_h, tgt_w), mode='bilinear', align_corners=False)
        feats = self.backbone(img)
        multi = self.decoder(feats)
        return self.query_net(multi)

