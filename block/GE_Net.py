#luohaozhe@stu.scu.edu.cn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
class Downblock(nn.Module):
    def __init__(self, channels, kernel_size=3, relu=True, stride=2, padding=1):
        super(Downblock, self).__init__()

        self.conv = nn.Conv2d(channels, channels, groups=channels, stride=stride,
                                kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = F.relu(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,padding=0):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GEBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, spatial, extent=2, extra_params=True, mlp=True, dropRate=0.0):
        # If extent is zero, assuming global.

        super(GEBlock, self).__init__()
        self.expansion=4
        channel = out_planes // self.expansion
        self.conv1 = BasicConv(in_planes, channel, kernel_size=1, stride=1)
        self.conv2 = BasicConv(channel, channel, kernel_size=3, stride=stride,padding=1)
        self.conv3 = BasicConv(channel, out_planes, kernel_size=1, stride=1)

        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut =nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,padding=0, bias=False) if (not self.equalInOut)  else None
        self.extent = extent

        if extra_params is True:
            if extent == 0:
                # Global DW Conv + BN
                self.downop = Downblock(out_planes, relu=False, kernel_size=spatial, stride=1, padding=0)
            elif extent == 2:
                self.downop = Downblock(out_planes, relu=False)

            elif extent == 4:
                self.downop = nn.Sequential(Downblock(out_planes, relu=True),
                                            Downblock(out_planes, relu=False))
            elif extent == 8:
                self.downop = nn.Sequential(Downblock(out_planes, relu=True),
                                            Downblock(out_planes, relu=True),
                                            Downblock(out_planes, relu=False))

            else:

                raise NotImplementedError('Extent must be 0,2,4 or 8 for now')

        else:
            if extent == 0:
                self.downop = nn.AdaptiveAvgPool2d(1)

            else:
                self.downop = nn.AdaptiveAvgPool2d(spatial // extent)

        if mlp is True:
            self.mlp = nn.Sequential(nn.Conv2d(out_planes, out_planes // 16, kernel_size=1, padding=0, bias=False),
                                     nn.ReLU(),
                                     nn.Conv2d(out_planes // 16, out_planes, kernel_size=1, padding=0, bias=False),
                                     )
        else:
            self.mlp = Identity()

    def forward(self, x):
        identity=x
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.conv3(out)
        shape_in = out.shape[-1]
        map = self.downop(out)
        map = self.mlp(map)
        map = F.interpolate(map, shape_in)
        map = torch.sigmoid(map)
        out = out * map

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


if __name__ == "__main__":
    input = torch.randn(1,3, 256, 256)
    net = GEBlock(3,20,2,64)
    device = torch.device("cuda")
    net.to(device)
    input = input.to(device)
    out = net(input)
    print(out.shape)