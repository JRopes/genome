# Code taken from: https://towardsdev.com/original-u-net-in-pytorch-ebe7bb705cc7
# You can use a pretrained model as well: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
import imageio as imageio
import torch
import torch.nn as nn
import torchvision as torchvision
from PIL import Image
import matplotlib.pyplot as plt

from DownConvolution import DownConvolution
from UpConvolution import UpConvolution
from LastConvolution import LastConvolution
from SimpleConvolution import SimpleConvolution
from utils import crop_img


class UNet(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(UNet, self).__init__()
        self.simpleConv = SimpleConvolution(input_channel, 64)
        self.downConvBlock1 = DownConvolution(64, 128)
        self.downConvBlock2 = DownConvolution(128, 256)
        self.downConvBlock3 = DownConvolution(256, 512)
        self.midMaxpool = nn.MaxPool2d(2, 2)
        self.bridge = UpConvolution(512, 1024)
        self.upConvBlock1 = UpConvolution(1024, 512)
        self.upConvBlock2 = UpConvolution(512, 256)
        self.upConvBlock3 = UpConvolution(256, 128)
        self.lastConv = LastConvolution(128, 64, num_classes)

    def forward(self, x):
        x_1 = self.simpleConv(x)
        x_2 = self.downConvBlock1(x_1)
        x_3 = self.downConvBlock2(x_2)
        x_4 = self.downConvBlock3(x_3)
        x_5 = self.midMaxpool(x_4)
        x_6 = self.bridge(x_5)
        crop_x_4 = crop_img(x_4, x_6)
        concat_x_4_6 = torch.cat((crop_x_4, x_6), 1)
        x_7 = self.upConvBlock1(concat_x_4_6)
        crop_x_3 = crop_img(x_3, x_7)
        concat_x_3_7 = torch.cat((crop_x_3, x_7), 1)
        x_8 = self.upConvBlock2(concat_x_3_7)
        crop_x_2 = crop_img(x_2, x_8)
        concat_x_2_8 = torch.cat((crop_x_2, x_8), 1)
        x_9 = self.upConvBlock3(concat_x_2_8)
        crop_x_1 = crop_img(x_1, x_9)
        concat_x_1_9 = torch.cat((crop_x_1, x_9), 1)
        out = self.lastConv(concat_x_1_9)

        return out


if __name__ == "__main__":
    unet = UNet(1, 2)
    image_path = "./TestData/BraTS2021_00000_t1ce_slice80.png"
    inp = Image.open(image_path)
    print("Size of Original Image:", inp.size)
    transform = torchvision.transforms.Resize((572, 572))
    print("Resized image:", inp.size)
    inp = transform(inp)
    plt.imshow(inp)
    plt.show()
    # inp = imageio.imread(image_path)
    inpt = torch.rand(1, 1, 572, 572)
    out = unet(inpt)
    # Must output (batch_size, num_classes, w, h)
    # (4, 2, 388, 388)
    # print(out.size())


