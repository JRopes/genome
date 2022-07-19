import torch.nn as nn

from .base import *


class BaselineShapeSegmenter(nn.Module):
    def __init__(self, in_channels=2):
        super(BaselineShapeSegmenter, self).__init__()
        print(in_channels)
        # Down sampling
        model = [
            Conv(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
                activation=nn.LeakyReLU(),
            )
        ]
        model += [
            Conv(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                activation=nn.LeakyReLU(),
            )
        ]
        model += [
            Conv(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                activation=nn.LeakyReLU(),
            )
        ]
        model += [
            Conv(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                activation=nn.LeakyReLU(),
            )
        ]

        for _ in range(3):
            model += [SN_ResidualBlock(256, activation=nn.LeakyReLU())]

        # Upsampling
        model += [
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(inplace=True),
        ]
        model += [
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),
        ]
        model += [
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(inplace=True),
        ]

        model += [
            Conv(
                in_channels=32,
                out_channels=1,
                kernel_size=5,
                stride=1,
                padding=2,
                activation=nn.LeakyReLU(),
            )
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class BaselineGradeSegmenter(nn.Module):
    def __init__(self, in_channels=2):
        super(BaselineGradeSegmenter, self).__init__()
        # Down sampling
        model = [
            Conv(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
                activation=nn.LeakyReLU(),
            )
        ]
        model += [
            Conv(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                activation=nn.LeakyReLU(),
            )
        ]
        model += [
            Conv(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                activation=nn.LeakyReLU(),
            )
        ]
        model += [
            Conv(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                activation=nn.LeakyReLU(),
            )
        ]
        for _ in range(3):
            model += [SN_ResidualBlock(256, activation=nn.LeakyReLU())]
        # Upsampling
        model += [
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(inplace=True),
        ]
        model += [
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),
        ]
        model += [
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(inplace=True),
        ]
        model += [
            Conv(
                in_channels=32,
                out_channels=1,
                kernel_size=5,
                stride=1,
                padding=2,
                activation=nn.LeakyReLU(),
            )
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x
