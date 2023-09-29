from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cfg


class SSDLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = cfg.default_boxes.alpha
        self.ratio_pos_neg = cfg.default_boxes.ratio_pos_neg
        self.label_smooth = cfg.default_boxes.label_smooth

    def forward(self, targets: Tuple[torch.Tensor, torch.Tensor], predictions: Tuple[torch.Tensor, torch.Tensor]):
        gt_bboxes, gt_labels = targets
        pred_bboxes, pred_labels = predictions
        # Number of positive
        pos_mask = gt_labels > 0
        num_pos = pos_mask.sum(1, keepdim=True).sum()
        # Numer of negative
        num_neg = pos_mask.sum(1, keepdim=True) * self.ratio_pos_neg
        box_loss = F.smooth_l1_loss(pred_bboxes[pos_mask], gt_bboxes[pos_mask], reduction='sum')
        conf_loss = F.cross_entropy(pred_labels.view(-1, cfg.voc_dataset.num_classes), 
                                    gt_labels.view(-1),
                                    reduction='none').view(gt_labels.size())
        # Hard negative mining
        neg_loss = conf_loss.clone()
        neg_loss[pos_mask] = -float('inf')
        _, neg_idx = neg_loss.sort(1, descending=True)
        background_idxs = neg_idx.sort(1)[1] < num_neg
        num_pos = max(1, num_pos)
        # Total losses
        reg_loss = box_loss.sum() / num_pos
        cls_loss = (conf_loss[pos_mask].sum() + conf_loss[background_idxs].sum()) / num_pos
        
        return reg_loss, cls_loss