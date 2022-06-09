import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if "multi_channel" in kwargs:
            self.multi_channel = kwargs["multi_channel"]
            kwargs.pop("multi_channel")
        else:
            self.multi_channel = False

        if "return_mask" in kwargs:
            self.return_mask = kwargs["return_mask"]
            kwargs.pop("return_mask")
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        else:
            self.weight_maskUpdater = torch.ones(
                1, 1, self.kernel_size[0], self.kernel_size[1]
            )

        self.slide_winsize = (
            self.weight_maskUpdater.shape[1]
            * self.weight_maskUpdater.shape[2]
            * self.weight_maskUpdater.shape[3]
        )

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(
                            input.data.shape[0],
                            input.data.shape[1],
                            input.data.shape[2],
                            input.data.shape[3],
                        ).to(input)
                    else:
                        mask = torch.ones(
                            1, 1, input.data.shape[2], input.data.shape[3]
                        ).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
        #     self.update_mask.to(input)
        #     self.mask_ratio.to(input)

        raw_out = super(PartialConv2d, self).forward(
            torch.mul(input, mask_in) if mask_in is not None else input
        )

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class PartialConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        activation=nn.LeakyReLU(inplace=True),
    ):
        super(PartialConv, self).__init__()
        if activation:
            self.conv = nn.Sequential(
                PartialConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.InstanceNorm2d(out_channels),
                activation,
            )
        else:
            self.conv = nn.Sequential(
                PartialConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.InstanceNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class SN_Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        activation=nn.ReLU(inplace=True),
    ):
        super(SN_Conv, self).__init__()
        self.SpectralNorm = torch.nn.utils.spectral_norm
        if activation:
            self.conv = nn.Sequential(
                self.SpectralNorm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    )
                ),
                nn.InstanceNorm2d(out_channels),
                activation,
            )
        else:
            self.conv = nn.Sequential(
                self.SpectralNorm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    )
                ),
                nn.InstanceNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class SN_ResidualBlock(nn.Module):
    def __init__(
        self,
        in_features,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        activation=nn.LeakyReLU(inplace=True),
    ):
        super(SN_ResidualBlock, self).__init__()
        conv_block = [
            SN_Conv(
                in_features,
                in_features,
                kernel_size,
                stride,
                padding,
                dilation,
                activation=activation,
            ),
            SN_Conv(in_features, in_features, activation=False),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        activation=nn.ReLU(inplace=True),
    ):
        super(Conv, self).__init__()
        if activation:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.InstanceNorm2d(out_channels),
                activation,
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.InstanceNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        activation=nn.ReLU(inplace=True),
    ):
        super(ResidualBlock, self).__init__()

        conv_block = [
            Conv(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                activation=activation,
            ),
            Conv(in_channels, in_channels, activation=False),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class SN_ResidualBlock(nn.Module):
    def __init__(
        self,
        in_features,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        activation=nn.LeakyReLU(inplace=True),
    ):
        super(SN_ResidualBlock, self).__init__()
        conv_block = [
            SN_Conv(
                in_features,
                in_features,
                kernel_size,
                stride,
                padding,
                dilation,
                activation=activation,
            ),
            SN_Conv(in_features, in_features, activation=False),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Partial_ResidualBlock(nn.Module):
    def __init__(
        self,
        in_features,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        activation=nn.ReLU(inplace=True),
    ):
        super(Partial_ResidualBlock, self).__init__()
        conv_block = [
            PartialConv(
                in_features,
                in_features,
                kernel_size,
                stride,
                padding,
                dilation,
                activation=activation,
            ),
            PartialConv(in_features, in_features, activation=False),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
