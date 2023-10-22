from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cfg

class IoULoss:
    @classmethod
    def diou_loss(cls, target_bboxes, pred_bboxes):
        """Reference: https://arxiv.org/pdf/1911.08287.pdf
        Args:
            target_bboxes (torch.Tensor): Target boundong boxes, [N, H, W, 4]
            pred_bboxes (torch.Tensor): Predicted bounding boxes, [N, H,W, 4]
        """
        # Compute intersections
        x1 = torch.max(target_bboxes[..., 0], pred_bboxes[..., 0])
        y1 = torch.max(target_bboxes[..., 1], pred_bboxes[..., 1])
        x2 = torch.min(target_bboxes[..., 2], pred_bboxes[..., 2])
        y2 = torch.min(target_bboxes[..., 3], pred_bboxes[..., 3])

        intersects = torch.clamp((x2-x1), 0) * torch.clamp((y2-y1), 0)

        # Compute unions
        A = abs((target_bboxes[..., 2]-target_bboxes[..., 0]) * (target_bboxes[..., 3]-target_bboxes[..., 1]))
        B = abs((pred_bboxes[..., 2]-pred_bboxes[..., 0]) * (pred_bboxes[..., 3]-pred_bboxes[..., 1]))

        unions = A + B - intersects
        ious = intersects / unions
        
        cx1 = torch.min(target_bboxes[..., 0], pred_bboxes[..., 0])
        cy1 = torch.min(target_bboxes[..., 1], pred_bboxes[..., 1])
        cx2 = torch.max(target_bboxes[..., 2], pred_bboxes[..., 2])
        cy2 = torch.max(target_bboxes[..., 3], pred_bboxes[..., 3])
        
        c_dist = ((target_bboxes[..., 2] + target_bboxes[..., 0] - pred_bboxes[..., 2] - pred_bboxes[..., 0]) ** 2 + \
                (target_bboxes[..., 3] + target_bboxes[..., 1] - pred_bboxes[..., 3] - pred_bboxes[..., 1]) ** 2) / 4
        diagonal_l2 = (cx2-cx1) **2 + (cy2-cy1) ** 2 
        
        return ious - c_dist / diagonal_l2

    @classmethod
    def giou_loss(target_bboxes, pred_bboxes):
        """Reference: https://arxiv.org/abs/1902.09630

        Args:
            target_bboxes (torch.Tensor): Target boundong boxes, [N, H, W, 4]
            pred_bboxes (torch.Tensor): Predicted bounding boxes, [N, H,W, 4]
        """
        # Compute intersection
        x1 = torch.max(target_bboxes[..., 0], pred_bboxes[..., 0])
        y1 = torch.max(target_bboxes[..., 1], pred_bboxes[..., 1])
        x2 = torch.min(target_bboxes[..., 2], pred_bboxes[..., 2])
        y2 = torch.min(target_bboxes[..., 3], pred_bboxes[..., 3])

        intersections = torch.clamp(x2-x1, 0) * torch.clamp(y2-y1, 0)

        # Compute Union
        A = abs((target_bboxes[..., 2]-target_bboxes[..., 0]) * (target_bboxes[..., 3]-target_bboxes[..., 1]))
        B = abs((pred_bboxes[..., 2]-pred_bboxes[..., 0]) * (pred_bboxes[..., 3]-pred_bboxes[..., 1]))

        unions = A + B - intersections
        iou = intersections / unions

        cx1 = torch.min(target_bboxes[..., 0], pred_bboxes[..., 0])
        cy1 = torch.min(target_bboxes[..., 1], pred_bboxes[..., 1])
        cx2 = torch.max(target_bboxes[..., 2], pred_bboxes[..., 2])
        cy2 = torch.max(target_bboxes[..., 3], pred_bboxes[..., 3])

        C = (cx2 - cx1) * (cy2 - cy1)

        return iou-(C-unions)/C


class SSDLoss(nn.Module):
    def __init__(self, giou=False, diou=False) -> None:
        super().__init__()
        self.giou = giou
        self.diou = diou
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

        if self.giou:
            box_loss = 1 - IoULoss.giou_loss(gt_bboxes[pos_mask], pred_bboxes[pos_mask])
        elif self.diou:
            box_loss = 1 - IoULoss.diou_loss(gt_bboxes[pos_mask], pred_bboxes[pos_mask])
        else:
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