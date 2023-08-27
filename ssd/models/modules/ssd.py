from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .neck import SSDNeck
from .head import SSDHead

from . import cfg


class SSDModel(nn.Module):
    def __init__(self, arch_name='vgg16', pretrained=True) -> None:
        super().__init__()
        self.extract_features, feat_dims = build_backbone(arch_name, pretrained)
        self.neck = SSDNeck(feat_dims)
        self.head = SSDHead(cfg.dataset.num_classes, cfg.models.fm_channels, cfg.default_boxes.dfboxes_sizes)
        
    def forward(self, x):
        x = self.extract_features(x)
        x = self.neck(x)
        x = self.head(x)
        return x