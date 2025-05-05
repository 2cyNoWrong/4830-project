import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthDecoder(nn.Module):
    """
    Task-specified decoder for depth map generation:
      - Deconvolution (ConvTranspose2d) for upsampling
      - Convolution for feature refinement
      - Bilinear upsampling to match input resolution
    Processes multi-scale feature maps and outputs a depth map matching input RGB size.
    """
    def __init__(self, encoder_dims, mid_channels=256, out_channels=1):
        super().__init__()
        self.deconv_blocks = nn.ModuleList()
        self.conv_blocks   = nn.ModuleList()
        for in_c in reversed(encoder_dims):
            self.deconv_blocks.append(
                nn.ConvTranspose2d(in_c, mid_channels,
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1)
            )
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels,
                              kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.final_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, feats, target_size=None):
        """
        Args:
            feats (List[Tensor]): multi-scale feature maps
            target_size (tuple): desired output (H, W)
        """
        x = None
        # Upsample without skip connections
        for deconv, conv, feat in zip(
                self.deconv_blocks, self.conv_blocks, reversed(feats)):
            inp = feat if x is None else x
            x = deconv(inp)
            x = conv(x)
        # Final bilinear upsample
        x = F.interpolate(x, scale_factor=2,
                          mode='bilinear', align_corners=False)
        depth = self.final_conv(x)
        if target_size is not None and depth.shape[-2:] != target_size:
            depth = F.interpolate(depth, size=target_size,
                                  mode='bilinear', align_corners=False)
        return depth

# Integration example in DepthEstimationModel:

# --------------------------------------------------------------------
# from model.DepthDecoder import DepthDecoder
# self.depth_decoder = DepthDecoder([192,384,768,1536], mid_channels=256, out_channels=1)
#
# def forward(self, imgs: torch.Tensor, ...):
#     feats = self.seecoder.backbone(imgs)
#     depth_map = self.depth_decoder(feats)
#     assert depth_map.shape[-2:] == imgs.shape[-2:]
#     return depth_map
# --------------------------------------------------------------------
