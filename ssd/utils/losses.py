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
        self.ratio_pos = cfg.default_boxes.pos_ratio
        self.ratio_neg = cfg.default_boxes.neg_ratio
        self.alpha = cfg.default_boxes.alpha
        self.label_smooth = cfg.default_boxes.label_smooth
    
    def forward(self, targets: Tuple[torch.Tensor, torch.Tensor], predictions: Tuple[torch.Tensor, torch.Tensor]):
        gt_bboxes, gt_labels = targets
        pred_bboxes, pred_labels = predictions
        gt_bboxes, gt_labels = gt_bboxes.reshape(-1, 4), gt_labels.reshape(-1, 1)
        pred_bboxes, pred_labels = pred_bboxes.reshape(-1, 4), pred_labels.reshape(-1, 1)

        pos_mask = gt_bboxes.sum(dim=1)>0
        N = max(1, len(pos_mask))

        pos_gt_labels = gt_labels[pos_mask]
        neg_gt_labels = gt_labels[~pos_mask]

        pos_pred_labels = pred_labels[pos_mask]
        neg_pred_labels = pred_labels[~pos_mask]

        loc_loss = F.smooth_l1_loss(gt_bboxes[pos_mask], pred_bboxes[pos_mask], reduction='mean')
        pos_conf_loss = F.cross_entropy(pos_gt_labels, pos_pred_labels, reduction='mean', label_smoothing=self.label_smooth)
        neg_conf_loss = F.cross_entropy(neg_gt_labels, neg_pred_labels, reduction='none', label_smoothing=self.label_smooth)

        top_neg_con_loss = self.hard_negative_mining(N, neg_conf_loss).mean()    
        conf_loss =  self.alpha * (pos_conf_loss + top_neg_con_loss) / N
        return loc_loss, conf_loss

    def hard_negative_mining(self, num_pos, conf_losses):
        sorted_conf_losses = torch.sort(conf_losses, dim=0)[0]
        top_neg_conf_loss = sorted_conf_losses[:num_pos*self.ratio_pos]
        return top_neg_conf_loss
        
