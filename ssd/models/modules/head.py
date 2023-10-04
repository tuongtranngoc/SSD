from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
import torch.nn as nn

from . import *


class SSDHead(nn.Module):
    def __init__(self, num_classes, in_channels, out_channels) -> None:
        super().__init__()
        self.cls_confs = nn.ModuleList()
        self.reg_boxes = nn.ModuleList()
        
        for num_in, num_out in zip(in_channels, out_channels):
            self.cls_confs.append(
                nn.Conv2d(num_in, num_classes * num_out, kernel_size=3, padding=1, stride=1))
            self.reg_boxes.append(
                nn.Conv2d(num_in, num_out * 4, kernel_size=3, padding=1, stride=1))
        
        xavier_init(self.cls_confs)
        xavier_init(self.reg_boxes)

    def _get_result_from_module_list(self, x: torch.Tensor, idx: int, module_list: nn.Module) -> torch.Tensor:
        num_blocks = len(module_list)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in module_list:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x):
        cls_results = []
        reg_results = []
        for i, input in enumerate(x):
            cls_out = self._get_result_from_module_list(input, i, self.cls_confs)
            reg_out = self._get_result_from_module_list(input, i, self.reg_boxes)
            
            N, _, H, W = cls_out.shape
            
            cls_out = cls_out.view(N, -1, cfg.voc_dataset.num_classes, H, W)
            cls_out = cls_out.permute(0, 3, 4, 1, 2)
            cls_out = cls_out.reshape(N, -1, cfg.voc_dataset.num_classes)

            reg_out = reg_out.view(N, -1, 4, H, W)
            reg_out = reg_out.permute(0, 3, 4, 1, 2)
            reg_out = reg_out.reshape(N, -1, 4)
            
            reg_results.append(reg_out)
            cls_results.append(cls_out)
            
        cls_results = torch.cat(cls_results, dim=1)
        reg_results = torch.cat(reg_results, dim=1)
        
        return reg_results, cls_results