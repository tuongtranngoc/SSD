from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from typing import Tuple, List

import torch
import torch.nn as nn


class SSDLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.location_loss = nn.SmoothL1Loss()
        self.confidence_loss = nn.CrossEntropyLoss()
        self.ratio = [3, 1]

    def compute_loss(self, targets: Tuple[torch.Tensor, torch.Tensor], predictions: Tuple[torch.Tensor, torch.Tensor]):
        gt_bboxes, gt_labels = targets
        pred_bboxes, pred_labels = predictions
        pos_idxs = gt_bboxes[gt_bboxes.sum(dim=1) > 0]

        loc_loss = self.location_loss(gt_bboxes[pos_idxs], pred_bboxes[pos_idxs])

    def hard_negative_mining(self, pos_idxs, gt_labels, pred_labels):
        pos_gt_labels = gt_labels[pos_idxs]
        neg_gt_labels = gt_labels[~pos_idxs]

        pos_pred_labels = pred_labels[pos_idxs]
        neg_pred_labels = pred_labels[~pos_idxs]

        neg_pred_labels = torch.sort()
