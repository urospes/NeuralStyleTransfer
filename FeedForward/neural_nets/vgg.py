from collections import namedtuple

import torch
from torchvision import models


class Vgg16(torch.nn.Module):

    def __init__(self):
        super(Vgg16, self).__init__()

        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        self.blk1 = torch.nn.Sequential()
        self.blk2 = torch.nn.Sequential()
        self.blk3 = torch.nn.Sequential()
        self.blk4 = torch.nn.Sequential()

        for i in range(4):
            self.blk1.add_module(str(i), vgg16[i])
        for i in range(4, 9):
            self.blk2.add_module(str(i), vgg16[i])
        for i in range(9, 16):
            self.blk3.add_module(str(i), vgg16[i])
        for i in range(16, 23):
            self.blk4.add_module(str(i), vgg16[i])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input):
        out = self.blk1(input)
        relu_1_2 = out
        out = self.blk2(out)
        relu_2_2 = out
        out = self.blk3(out)
        relu_3_3 = out
        out = self.blk4(out)
        relu_4_3 = out

        vgg_outputs = namedtuple("vgg_outputs", ["relu_1_2", "relu_2_2", "relu_3_3", "relu_4_3"])
        return vgg_outputs(relu_1_2=relu_1_2, relu_2_2=relu_2_2, relu_3_3=relu_3_3, relu_4_3=relu_4_3)
