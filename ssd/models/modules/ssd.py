from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import vgg16_extractor
from .neck import SSDNeck
from .head import SSDHead

from . import cfg


class SSDModel(nn.Module):
    def __init__(self, arch_name) -> None:
        super().__init__()
        backbone = vgg16_extractor(arch_name)
        self.neck = SSDNeck(backbone)
        self.head = SSDHead(cfg.voc_dataset.num_classes, cfg.models.fm_channels, cfg.default_boxes.num_dfboxes)
    
    def forward(self, x):
        x = self.neck(x)
        x = self.head(x)
        return x