from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models

from .base import *


class BaselineInpaintingGenerator(nn.Module):
    def __init__(self):
        super(BaselineInpaintingGenerator, self).__init__()

        model = [
            Conv(in_channels=5, out_channels=32, kernel_size=3, stride=2, padding=1),
            Conv(in_channels=32, out_channels=128, kernel_size=3, stride=2, padding=1),
        ]
        for _ in range(3):
            model += [ResidualBlock(128)]
        self.model_initial = nn.Sequential(*model)

        model = [
            Conv(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1),
            Conv(in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1),
        ]
        self.model_Mask = nn.Sequential(*model)

        model = [
            Conv(in_channels=144, out_channels=256, kernel_size=3, stride=1, padding=1),
            ResidualBlock(256),
            nn.Upsample(scale_factor=2, mode="nearest"),
            Conv(256, 128, 3, 1, 1),
            ResidualBlock(128),
            nn.Upsample(scale_factor=2, mode="nearest"),
            Conv(128, 64, 3, 1, 1),
            ResidualBlock(64),
            Conv(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3),
        ]
        self.model_flair = nn.Sequential(*model)

        model = [
            Conv(in_channels=144, out_channels=256, kernel_size=3, stride=1, padding=1),
            ResidualBlock(256),
            nn.Upsample(scale_factor=2, mode="nearest"),
            Conv(256, 128, 3, 1, 1),
            ResidualBlock(128),
            nn.Upsample(scale_factor=2, mode="nearest"),
            Conv(128, 64, 3, 1, 1),
            ResidualBlock(64),
            Conv(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3),
        ]
        self.model_t1 = nn.Sequential(*model)

        model = [
            Conv(in_channels=144, out_channels=256, kernel_size=3, stride=1, padding=1),
            ResidualBlock(256),
            nn.Upsample(scale_factor=2, mode="nearest"),
            Conv(256, 128, 3, 1, 1),
            ResidualBlock(128),
            nn.Upsample(scale_factor=2, mode="nearest"),
            Conv(128, 64, 3, 1, 1),
            ResidualBlock(64),
            Conv(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3),
        ]
        self.model_t1ce = nn.Sequential(*model)

        model = [
            Conv(in_channels=144, out_channels=256, kernel_size=3, stride=1, padding=1),
            ResidualBlock(256),
            nn.Upsample(scale_factor=2, mode="nearest"),
            Conv(256, 128, 3, 1, 1),
            ResidualBlock(128),
            nn.Upsample(scale_factor=2, mode="nearest"),
            Conv(128, 64, 3, 1, 1),
            ResidualBlock(64),
            Conv(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3),
        ]
        self.model_t2 = nn.Sequential(*model)

    def forward(self, brain_blank, M):
        x = self.model_initial(torch.cat((brain_blank, M), 1))
        M = self.model_Mask(M)
        x = torch.cat((x, M), 1)
        t1, t1ce, t2, flair = (
            self.model_t1(x),
            self.model_t1ce(x),
            self.model_t2(x),
            self.model_flair(x),
        )
        return torch.cat((t1, t1ce, t2, flair), 1)


class BaselineInpaintingDiscriminator(nn.Module):
    def __init__(self, in_channels=5, img_size=240):
        super(BaselineInpaintingDiscriminator, self).__init__()

        model = [
            SN_Conv(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        ]
        model += [
            SN_Conv(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        ]
        model += [
            SN_Conv(
                in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1
            )
        ]
        for _ in range(3):
            model += [SN_ResidualBlock(256)]
        model += [
            SN_Conv(
                in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1
            )
        ]
        model += [
            SN_Conv(
                in_channels=64,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.Sigmoid(),
            )
        ]

        self.model = nn.Sequential(*model)
        self.Avg = nn.AvgPool2d(kernel_size=img_size // 6)

    def forward(self, tumor, M):
        x = self.model(torch.cat((tumor, M), 1))
        return self.Avg(x).view(x.size()[0], -1)


class BaselineVGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(BaselineVGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select = ["0", "5", "10", "19", "28"]
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features
