# https://github.com/milesial/Pytorch-UNet
from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinCosEncoder(nn.Module):

    def __init__(self, out_features):
        super().__init__()
        self.num_components = out_features // 2
        self.in_features = 1
        self.out_features = self.in_features * 2 * self.num_components
        
        self.register_buffer("freqs", 2 ** torch.arange(self.num_components) * pi)

    def forward(self, x):
        aux = self.freqs[None, :] * x[:, None]
        sincos = torch.cat([torch.sin(aux), torch.cos(aux)], dim=-1)
        return sincos


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, time_embedding_dim, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

        self.time_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(time_embedding_dim, out_channels)
        )

    def forward(self, x, time_emb):
        x = self.conv1(x)
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        x = x + time_emb
        return self.conv2(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, time_embedding_dim)

    def forward(self, x, time_emb):
        x = self.maxpool(x)
        return self.conv(x, time_emb)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, time_embedding_dim, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, time_embedding_dim, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, time_embedding_dim)

    def forward(self, x1, x2, time_emb):
        x1 = self.up(x1)
        if x2 == None:
            return self.conv(x1)
        else:
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)

            return self.conv(x, time_emb)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, 
                 out_channels: int, 
                 time_embedding_dim: int, 
                 time_steps: int, 
                 channel_mult_factor: int = 1, 
                 bilinear=False):
        super(UNet, self).__init__()

        self.time_steps = time_steps
        self.encoder = SinCosEncoder(time_embedding_dim)

        self.first_layer_t = nn.Sequential(
            nn.Linear(self.encoder.out_features, time_embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim)
        )

        self.inc = DoubleConv(in_channels, int(64  * channel_mult_factor), time_embedding_dim)

        self.down1 = Down(int(64 * channel_mult_factor), int(128 * channel_mult_factor), time_embedding_dim)
        self.down2 = Down(int(128 * channel_mult_factor), int(256 * channel_mult_factor), time_embedding_dim)
        self.down3 = Down(int(256 * channel_mult_factor), int(512 * channel_mult_factor), time_embedding_dim)
        factor = 2 if bilinear else 1
        self.down4 = Down(int(512 * channel_mult_factor), int(1024 * channel_mult_factor) // factor, time_embedding_dim)
        self.up1 = Up(int(1024 * channel_mult_factor), int(512 * channel_mult_factor) // factor, time_embedding_dim, bilinear)
        self.up2 = Up(int(512 * channel_mult_factor), int(256 * channel_mult_factor) // factor, time_embedding_dim, bilinear)
        self.up3 = Up(int(256 * channel_mult_factor), int(128 * channel_mult_factor) // factor, time_embedding_dim, bilinear)
        self.up4 = Up(int(128 * channel_mult_factor), int(64 * channel_mult_factor), time_embedding_dim, bilinear)

        self.outc = OutConv(int(64 * channel_mult_factor), out_channels)

    def forward(self, image, t):
        x = image

        # Rescale t to the range [-1, 1]
        t = 2.0 * t / self.time_steps - 1.0
        # Positional encoding of timestep
        temb = self.encoder(t)
        temb = self.first_layer_t(temb)

        x1 = self.inc(x, temb)
        x2 = self.down1(x1, temb)
        x3 = self.down2(x2, temb)
        x4 = self.down3(x3, temb)
        x5 = self.down4(x4, temb)

        x = self.up1(x5, x4, temb)
        x = self.up2(x, x3, temb)
        x = self.up3(x, x2, temb)
        x = self.up4(x, x1, temb)
        logits = self.outc(x)

        return logits