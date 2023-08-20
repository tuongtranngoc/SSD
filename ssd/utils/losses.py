from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
import torch.nn as nn


class SSDLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def compute_loss(target, predtions, defaultboxes):
        pass

    