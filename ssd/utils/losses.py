from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import *

class IoULoss:
    dfboxes = DefaultBoxesGenerator.df_bboxes.to(cfg.device)

    @classmethod
    def diou_loss(cls, target_bboxes, pred_bboxes):
        """Reference: https://arxiv.org/pdf/1911.08287.pdf
        Args:
            target_bboxes (torch.Tensor): Target boundong boxes, [N, H, W, 4]
            pred_bboxes (torch.Tensor): Predicted bounding boxes, [N, H,W, 4]
        """
        target_bboxes = target_bboxes.clone()
        pred_bboxes = pred_bboxes.clone()
        
        target_bboxes = BoxUtils.decode_ssd(target_bboxes, cls.dfboxes)
        pred_bboxes = BoxUtils.decode_ssd(pred_bboxes, cls.dfboxes)
        target_bboxes = BoxUtils.xcycwh_to_xyxy(target_bboxes)
        pred_bboxes = BoxUtils.xcycwh_to_xyxy(pred_bboxes)

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
        diagonal_l2 = torch.clamp((cx2-cx1), min=0.0) **2 + torch.clamp((cy2-cy1), min=0.0) ** 2
        
        return ious - c_dist / diagonal_l2

    @classmethod
    def giou_loss(cls, target_bboxes, pred_bboxes):
        """Reference: https://arxiv.org/abs/1902.09630

        Args:
            target_bboxes (torch.Tensor): Target boundong boxes, [N, H, W, 4]
            pred_bboxes (torch.Tensor): Predicted bounding boxes, [N, H,W, 4]
        """
        # Compute intersection
        target_bboxes = target_bboxes.clone()
        pred_bboxes = pred_bboxes.clone()

        target_bboxes = BoxUtils.decode_ssd(target_bboxes, cls.dfboxes)
        pred_bboxes = BoxUtils.decode_ssd(pred_bboxes, cls.dfboxes)
        target_bboxes = BoxUtils.xcycwh_to_xyxy(target_bboxes)
        pred_bboxes = BoxUtils.xcycwh_to_xyxy(pred_bboxes)

        x1 = torch.max(target_bboxes[..., 0], pred_bboxes[..., 0])
        y1 = torch.max(target_bboxes[..., 1], pred_bboxes[..., 1])
        x2 = torch.min(target_bboxes[..., 2], pred_bboxes[..., 2])
        y2 = torch.min(target_bboxes[..., 3], pred_bboxes[..., 3])

        intersections = torch.clamp(x2-x1, 0) * torch.clamp(y2-y1, 0)

        # Compute Union
        A = abs((target_bboxes[..., 2]-target_bboxes[..., 0]) * (target_bboxes[..., 3]-target_bboxes[..., 1]))
        B = abs((pred_bboxes[..., 2]-pred_bboxes[..., 0]) * (pred_bboxes[..., 3]-pred_bboxes[..., 1]))

        unions = A + B - intersections
        ious = intersections / unions

        cx1 = torch.min(target_bboxes[..., 0], pred_bboxes[..., 0])
        cy1 = torch.min(target_bboxes[..., 1], pred_bboxes[..., 1])
        cx2 = torch.max(target_bboxes[..., 2], pred_bboxes[..., 2])
        cy2 = torch.max(target_bboxes[..., 3], pred_bboxes[..., 3])

        C = (cx2 - cx1) * (cy2 - cy1)

        return ious-(C-unions)/C


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
        num_pos = max(1, num_pos)
        
        if self.giou:
            box_loss = 1.0 - IoULoss.giou_loss(gt_bboxes, pred_bboxes)
            reg_loss = box_loss[pos_mask].sum() / num_pos
        elif self.diou:
            box_loss = 1.0 - IoULoss.diou_loss(gt_bboxes, pred_bboxes)
            reg_loss = box_loss[pos_mask].sum() / num_pos
        else:
            box_loss = F.smooth_l1_loss(pred_bboxes, gt_bboxes, reduction='sum')
            reg_loss = box_loss.sum() / num_pos
        
        conf_loss = F.cross_entropy(pred_labels.view(-1, cfg.voc_dataset.num_classes), 
                                    gt_labels.view(-1),
                                    reduction='none').view(gt_labels.size())
        
        # Hard negative mining
        neg_loss = conf_loss.clone()
        neg_loss[pos_mask] = -float('inf')
        _, neg_idx = neg_loss.sort(1, descending=True)
        background_idxs = neg_idx.sort(1)[1] < num_neg
        # Total losses
        cls_loss = (conf_loss[pos_mask].sum() + conf_loss[background_idxs].sum()) / num_pos
        
        return reg_loss, cls_loss