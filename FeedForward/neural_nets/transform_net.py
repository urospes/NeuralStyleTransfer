import torch


class ImageTransformNet(torch.nn.Module):

    def __init__(self):
        super(ImageTransformNet, self).__init__()

        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.norm1 = torch.nn.InstanceNorm2d(32, affine=True)   # u originalnom radu se koristi batch normalizacija, ali se Instance normalizacijom dobijaju bolji rezultati
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.norm2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.norm3 = torch.nn.InstanceNorm2d(128, affine=True)

        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        self.conv4 = UpsampleConv(128, 64, kernel_size=3, stride=1, upsample_factor=2)
        self.norm4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv5 = UpsampleConv(64, 32, kernel_size=3, stride=1, upsample_factor=2)
        self.norm5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv6 = ConvLayer(32, 3, kernel_size=9, stride=1)

        self.relu = torch.nn.ReLU()

    def forward(self, input):
        out = self.relu(self.norm1(self.conv1(input)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.relu(self.norm3(self.conv3(out)))
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.relu(self.norm4(self.conv4(out)))
        out = self.relu(self.norm5(self.conv5(out)))
        out = self.conv6(out)
        return out


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                    padding=kernel_size//2, padding_mode='reflect')

    def forward(self, x):
        return self.conv(x)


class UpsampleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample_factor):
        super(UpsampleConv, self).__init__()
        self.upsample_factor = upsample_factor
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=kernel_size//2, padding_mode="reflect")

    def forward(self, input):
        if self.upsample_factor > 1:
            input = torch.nn.functional.interpolate(input, mode='nearest', scale_factor=self.upsample_factor)
        return self.conv(input)


class ResidualBlock(torch.nn.Module):

    def __init__(self, n_filters):
        super(ResidualBlock, self).__init__()

        self.conv1 = ConvLayer(n_filters, n_filters, kernel_size=3, stride=1)
        self.norm1 = torch.nn.InstanceNorm2d(n_filters, affine=True)
        self.conv2 = ConvLayer(n_filters, n_filters, kernel_size=3, stride=1)
        self.norm2 = torch.nn.InstanceNorm2d(n_filters, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        res = input
        out = self.relu(self.norm1(self.conv1(input)))
        out = self.norm2(self.conv2(out))
        return out + res
