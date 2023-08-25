from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .neck import SSDNeck


class SSD(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extract_features, feat_dims = build_backbone()
        self.neck = SSDNeck(feat_dims)
        
    def forward(self, x):
        x = self.extract_features(x)
        x = self.neck(x)
        return x