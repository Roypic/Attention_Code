import torch.nn as nn
import torch
class SCSEModule(nn.Module):
    def __init__(self, ch, redu=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch,ch//redu,1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(ch//redu,ch,1),
                                 nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(ch,ch,1),
                                 nn.Sigmoid())
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)