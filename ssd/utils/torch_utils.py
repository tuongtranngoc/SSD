from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
from . import cfg

class BoxUtils:
    h, w = cfg.models.image_size, cfg.models.image_size

    @classmethod
    def xcycwh_to_xyxy(cls, bboxes:torch.Tensor):
        bboxes = bboxes.clone()
        x1y1 = bboxes[..., :2] - bboxes[..., 2:] / 2
        x2y2 = bboxes[..., :2] + bboxes[..., 2:] / 2

        x1y1x2y2 = torch.cat((x1y1, x2y2), dim=1)
        x1y1x2y2 = torch.clamp(x1y1x2y2, min=0, max=1.0)

        return x1y1x2y2
    
    @classmethod
    def pairwise_ious(cls, x:torch.Tensor, y:torch.Tensor):
        x1 = torch.max(x[:, None, 0], y[..., 0])
        y1 = torch.max(x[:, None, 1], y[..., 1])
        x2 = torch.min(x[:, None, 2], y[..., 2])
        y2 = torch.min(x[:, None, 3], y[..., 3])

        intersect = torch.clamp((x2-x1), 0) * torch.clamp((y2-y1), 0)
        unions = abs((x[:, None, 2] - x[:, None,  0]) * (x[:, None, 3] - x[:, None, 1])) + abs((y[..., 2] - y[..., 0]) * (y[..., 3] - y[..., 1])) - intersect
        intersect[intersect.gt(0)] = intersect[intersect.gt(0)] / unions[intersect.gt(0)]
        
        return intersect
    
    @classmethod
    def normalize_box(cls, bboxes:torch.Tensor):
        bboxes = bboxes.clone()
        bboxes[..., [0, 2]] /= cls.w
        bboxes[..., [1, 3]] /= cls.h
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]].clamp(min=0.0, max=1.0)
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]].clamp(min=0.0, max=1.0)
        return bboxes
    
    @classmethod
    def denormalize_box(cls, bboxes:torch.Tensor):
        bboxes = bboxes.clone()
        bboxes[..., [0, 2]] *= cls.w
        bboxes[..., [1, 3]] *= cls.h
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]].clamp(min=0.0, max=cls.w)
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]].clamp(min=0.0, max=cls.h)
        return bboxes
