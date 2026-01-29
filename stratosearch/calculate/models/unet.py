import torch
from torch import nn
from torch.nn import functional as F


FEATURES_BASE = [32, 64, 128]


class GroupNormReLU(nn.Module):
    def __init__(self, num_channels, groups=8):
        super().__init__()
        g = max(1, min(groups, num_channels))
        self.gn = nn.GroupNorm(num_groups=g, num_channels=num_channels)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.gn(x))

class SepConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_act = GroupNormReLU(out_ch)
    def forward(self, x):
        return self.bn_act(self.pw(self.dw(x)))

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, separable=False, groups=8):
        super().__init__()
        if separable:
            self.net = nn.Sequential(
                SepConv2d(in_ch, out_ch),
                SepConv2d(out_ch, out_ch)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                GroupNormReLU(out_ch, groups=groups),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                GroupNormReLU(out_ch, groups=groups)
            )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=6, features=None, separable=False, gn_groups=8):
        super().__init__()
        if features is None:
            features = FEATURES_BASE
        self.separable = separable
        self.gn_groups = gn_groups

        self.encoders = nn.ModuleList()
        ch = in_ch
        for f in features:
            self.encoders.append(DoubleConv(ch, f, separable=self.separable, groups=self.gn_groups))
            ch = f

        mid = features[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=2, dilation=2, bias=False),
            GroupNormReLU(mid, groups=self.gn_groups),
            nn.Conv2d(mid, mid*2, kernel_size=3, padding=4, dilation=4, bias=False),
            GroupNormReLU(mid*2, groups=self.gn_groups),
        )

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev_ch = mid*2
        for f in reversed(features):
            up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(prev_ch, f, 3, padding=1, bias=False),
                GroupNormReLU(f, groups=self.gn_groups)
            )
            self.upconvs.append(up)
            self.decoders.append(DoubleConv(f*2, f, separable=self.separable, groups=self.gn_groups))
            prev_ch = f

        self.final_conv = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)

        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = skips[-(i+1)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = self.decoders[i](torch.cat([skip, x], dim=1))

        return self.final_conv(x)
