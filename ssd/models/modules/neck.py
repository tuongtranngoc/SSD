from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn


class SSDNeck(nn.Module):
    def __init__(self, feat_dims) -> None:
        super().__init__()
        self.conv6 = nn.Conv2d(feat_dims, 1024, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1, padding=1)
        self.conv8_2 = nn.Conv2d(1024, 512, kernel_size=3, padding=0)
        self.conv9_2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv10_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, x):
        pass

    