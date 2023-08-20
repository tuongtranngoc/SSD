from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
import torch.nn as nn


class SSDLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def compute_loss(gt_bboxes: torch.Tensor, df_bboxes: torch.Tensor, 
                     class_ids: torch.Tensor, pred_bboxes: torch.Tensor):
        pass
    