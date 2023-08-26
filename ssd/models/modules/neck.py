from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSDNeck(nn.Module):
    def __init__(self, feat_dims) -> None:
        super().__init__()
        # Parameters used for L2 normalization + rescale
        self.scale_weight = nn.Parameter(torch.ones(512) * 20)

        # FC
        self.fc = nn.Sequential(
            # conv6(FC6)
            nn.Conv2d(feat_dims, 1024, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            # conv7(FC7)
            nn.Conv2d(1024, 1024, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True)
            )

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
                nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=1)
            )
        ])

        extra_feature_layers.insert(0, self.fc)
        self.extra_feature_layers = extra_feature_layers

    def forward(self, x):
        # L2 Normalization + rescale for conv4_3
        # Reference: L2 NORMALIZATION LAYER (https://arxiv.org/pdf/1506.04579.pdf)
        l2_x = self.scale_weight.view(1, -1, 1, 1) * F.normalize(x)
        y = [l2_x]
        z = [l2_x.size(1)]
        for sq in self.extra_feature_layers:
            x = sq(x)
            y.append(x)
            # import pdb; pdb.set_trace()
            z.append(x.size(1))
        return y, z
    