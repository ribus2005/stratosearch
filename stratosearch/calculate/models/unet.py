import torch
import torch.nn as nn
import torch.nn.functional as F


UNET_FEATURES = [32, 64, 128, 256]

class GroupNormReLU(nn.Module):
    def __init__(self, num_channels, groups=8):
        super().__init__()
        g = max(1, min(groups, num_channels))
        self.gn = nn.GroupNorm(num_groups=g, num_channels=num_channels)
        self.act = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.act(self.gn(x))

class SepConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, groups_gn=8):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = GroupNormReLU(out_ch, groups=groups_gn)
    def forward(self,x):
        return self.bn(self.pw(self.dw(x)))

class AnisoFirstBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups_gn=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3,11), padding=(1,5), bias=False),
            GroupNormReLU(out_ch, groups=groups_gn),
            nn.Conv2d(out_ch, out_ch, kernel_size=(11,3), padding=(5,1), bias=False),
            GroupNormReLU(out_ch, groups=groups_gn),
        )
    def forward(self,x): return self.net(x)

class AttentionGate(nn.Module):
    """Простой attention gate для skip-соединений (как в Attention U-Net)"""
    def __init__(self, in_ch, gating_ch, inter_ch):
        super().__init__()
        self.theta = nn.Conv2d(in_ch, inter_ch, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(gating_ch, inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
    def forward(self, x, g):
        # x: skip feature, g: gating (decoder feature)
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        add = self.act(theta_x + phi_g)
        psi = self.sig(self.psi(add))
        return x * psi

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling — расширяет receptive field без сильного даунсемплинга"""
    def __init__(self, in_ch, out_ch, rates=(1,2,4,8), groups_gn=8):
        super().__init__()
        self.convs = nn.ModuleList()
        for r in rates:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r, dilation=r, bias=False),
                GroupNormReLU(out_ch, groups=groups_gn)
            ))
        self.project = nn.Sequential(
            nn.Conv2d(len(rates)*out_ch, out_ch, kernel_size=1, bias=False),
            GroupNormReLU(out_ch, groups=groups_gn)
        )
    def forward(self,x):
        feats = [c(x) for c in self.convs]
        return self.project(torch.cat(feats, dim=1))

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=6, features=None, init_features=64, use_aniso_first=True, separable=False, gn_groups=8):
        super().__init__()
        if features is None:
            features = FEATURES
        if features is None:
            features = [init_features, init_features*2, init_features*4, init_features*8]

        self.use_aniso_first = use_aniso_first
        self.separable = separable
        self.gn_groups = gn_groups

        f0 = features[0]

        if use_aniso_first:
            self.encoder1 = AnisoFirstBlock(in_ch, f0, groups_gn=gn_groups)
        else:
            self.encoder1 = self._block(in_ch, f0, separable=separable, groups_gn=gn_groups)

        self.pool1 = nn.Conv2d(f0, features[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder2 = self._block(features[1], features[1], separable=separable, groups_gn=gn_groups)

        self.pool2 = nn.Conv2d(features[1], features[2], kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder3 = self._block(features[2], features[2], separable=separable, groups_gn=gn_groups)

        self.pool3 = nn.Conv2d(features[2], features[3], kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder4 = self._block(features[3], features[3], separable=separable, groups_gn=gn_groups)

        self.pool4 = nn.Conv2d(features[3], features[3] * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[3] * 2, features[3] * 2, kernel_size=3, padding=1, bias=False),
            GroupNormReLU(features[3] * 2, groups=gn_groups),
            nn.Dropout2d(p=0.2)
        )
        self.aspp = ASPP(features[3] * 2, features[3], rates=(1, 4, 8, 12), groups_gn=gn_groups)

        self.upconv4 = nn.ConvTranspose2d(features[3], features[3], kernel_size=2, stride=2)
        self.att4 = AttentionGate(in_ch=features[3], gating_ch=features[3], inter_ch=features[3] // 2)
        self.decoder4 = self._block(features[3] * 2, features[3], separable=separable, groups_gn=gn_groups)

        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.att3 = AttentionGate(in_ch=features[2], gating_ch=features[2], inter_ch=features[2] // 2)
        self.decoder3 = self._block(features[2] * 2, features[2], separable=separable, groups_gn=gn_groups)

        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.att2 = AttentionGate(in_ch=features[1], gating_ch=features[1], inter_ch=features[1] // 2)
        self.decoder2 = self._block(features[1] * 2, features[1], separable=separable, groups_gn=gn_groups)

        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.att1 = AttentionGate(in_ch=features[0], gating_ch=features[0], inter_ch=max(features[0] // 4, 1))
        self.decoder1 = self._block(features[0] * 2, features[0], separable=separable, groups_gn=gn_groups)

        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)

        self.decoder_dropout = nn.Dropout2d(p=0.1)

    @staticmethod
    def _block(in_ch, out_ch, separable=False, groups_gn=8):
        if separable:
            return nn.Sequential(
                SepConv2d(in_ch, out_ch, kernel_size=3, padding=1, groups_gn=groups_gn),
                SepConv2d(out_ch, out_ch, kernel_size=3, padding=1, groups_gn=groups_gn),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                GroupNormReLU(out_ch, groups=groups_gn),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                GroupNormReLU(out_ch, groups=groups_gn),
            )

    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)

        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)
        b = self.aspp(b)

        d4 = self.upconv4(b)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        e4_att = self.att4(e4, d4)
        d4 = self.decoder4(torch.cat([d4, e4_att], dim=1))
        # d4 = self.decoder_dropout(d4)

        d3 = self.upconv3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        e3_att = self.att3(e3, d3)
        d3 = self.decoder3(torch.cat([d3, e3_att], dim=1))
        # d3 = self.decoder_dropout(d3)

        d2 = self.upconv2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        e2_att = self.att2(e2, d2)
        d2 = self.decoder2(torch.cat([d2, e2_att], dim=1))
        # d2 = self.decoder_dropout(d2)

        d1 = self.upconv1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        e1_att = self.att1(e1, d1)
        d1 = self.decoder1(torch.cat([d1, e1_att], dim=1))
        # d1 = self.decoder_dropout(d1)

        logits = self.final_conv(d1)
        return logits
