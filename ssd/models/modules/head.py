from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
import torch.nn as nn


class SSDHead(nn.Module):
    def __init__(self, num_classes, dfboxes_list, channels_list) -> None:
        super().__init__()
        self.cls_confs = nn.ModuleList()
        self.reg_boxes = nn.ModuleList()
        for num_dfboxes, num_channels in zip(dfboxes_list, channels_list):
            self.cls_confs.append(
                nn.Conv2d(num_channels, num_classes * num_dfboxes, kernel_size=3, padding=1, stride=1)
            )
            self.reg_boxes.append(
                nn.Conv2d(num_channels, num_dfboxes * 4, kernel_size=3, padding=1, stride=1)
            )
        
    def forward(self, x):
        results = []
        