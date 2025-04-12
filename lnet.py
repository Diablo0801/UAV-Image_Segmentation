import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.theta = nn.Conv2d(in_channels, inter_channels, 1)
        self.phi = nn.Conv2d(gating_channels, inter_channels, 1)
        self.psi = nn.Conv2d(inter_channels, 1, 1)

    def forward(self, x, g):
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        f = F.relu(theta_x + phi_g)
        psi = torch.sigmoid(self.psi(f))
        return x * psi

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Conv2d(out_channels * 5, out_channels, 1)

    def forward(self, x):
        shape = x.shape[2:]
        p = self.pool(x)
        p = F.interpolate(p, size=shape, mode='bilinear')
        c = torch.cat([
            self.conv1(x), self.conv6(x), self.conv12(x), self.conv18(x), p
        ], dim=1)
        return F.relu(self.out(c))

class LNet(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.e1 = ResidualBlock(64, 64)
        self.e2 = ResidualBlock(64, 128)
        self.e3 = ResidualBlock(128, 256)
        self.e4 = ResidualBlock(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.aspp = ASPP(512, 512)

        self.up1 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.ag1 = AttentionGate(512, 512, 256)
        self.d1 = ResidualBlock(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.ag2 = AttentionGate(256, 256, 128)
        self.d2 = ResidualBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.ag3 = AttentionGate(128, 128, 64)
        self.d3 = ResidualBlock(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.ag4 = AttentionGate(64, 64, 32)
        self.d4 = ResidualBlock(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x = self.initial(x)
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        b = self.aspp(self.pool(e4))

        d1 = self.up1(b)
        d1 = torch.cat([d1, self.ag1(e4, d1)], dim=1)
        d1 = self.d1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, self.ag2(e3, d2)], dim=1)
        d2 = self.d2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, self.ag3(e2, d3)], dim=1)
        d3 = self.d3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([d4, self.ag4(e1, d4)], dim=1)
        d4 = self.d4(d4)

        return self.out(d4)