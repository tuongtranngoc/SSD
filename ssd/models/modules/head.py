from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
import torch.nn as nn


class SSDHead(nn.Module):
    def __init__(self, num_classes, out_channels) -> None:
        super().__init__()
        self.cls_confs = nn.ModuleList()
        self.reg_boxes = nn.ModuleList()
        self.out_channels = out_channels
        self.num_classes = num_classes

    def forward(self, x, y):
        for num_in, num_out in zip(y, self.out_channels):
            self.cls_confs.append(
                nn.Conv2d(num_in, self.num_classes * num_out, kernel_size=3, padding=1, stride=1)
            )
            self.reg_boxes.append(
                nn.Conv2d(num_in, num_out * 4, kernel_size=3, padding=1, stride=1)
            )
        
        results = []
        for input, cls_module, reg_module in zip(x, self.cls_confs, self.reg_boxes):
            cls_out = cls_module(input)
            reg_out = reg_module(input)



        