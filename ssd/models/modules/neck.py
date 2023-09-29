from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import *


class SSDNeck(nn.Module):
    def __init__(self, backbone) -> None:
        super().__init__()
        _, _, maxpool3_pos, maxpool4_pos, _ = (i for i, layer in enumerate(backbone) if isinstance(layer, nn.MaxPool2d))
        # Patch ceil_mode for maxpool3 to get the same WxH output sizes as the paper
        backbone[maxpool3_pos].ceil_mode = True
        self.features = nn.Sequential(*backbone[:maxpool4_pos])
        
        # Parameters used for L2 normalization + rescale
        self.scale_weight = nn.Parameter(torch.ones(cfg.models.fm_channels[0]) * 15)
        
        # FC
        fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),  # add modified maxpool5
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6),  # FC6 with atrous
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),  # FC7
            nn.ReLU(inplace=True)
        )

        xavier_init(fc)
        
        # Feature extra layers
        extra_feature_layers = nn.ModuleList([
            nn.Sequential(
                # conv8_2
                nn.Conv2d(1024, 256, kernel_size=1, padding=0, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                # conv9_2
                nn.Conv2d(512, 128, kernel_size=1, padding=0, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                # conv10_2
                nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                # conv11_2
                nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=1),
                nn.ReLU(inplace=True)
            )
        ])

        xavier_init(extra_feature_layers)
        
        extra_feature_layers.insert(0, nn.Sequential(*backbone[maxpool4_pos:-1], fc)) # until conv5_3, skip maxpool5
        self.extra_feature_layers = extra_feature_layers
    
    def forward(self, x):
        x = self.features(x)
        # L2 Normalization + rescale for conv4_3
        # Reference: L2 NORMALIZATION LAYER in https://arxiv.org/pdf/1506.04579.pdf
        l2_x = self.scale_weight.view(1, -1, 1, 1) * F.normalize(x)
        y = [l2_x]
        for sq in self.extra_feature_layers:
            x = sq(x)
            y.append(x)
        return y