from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch

class BoxUtils:

    @classmethod
    def xcycwh_to_xyxy(cls, bboxes, max_size):
        bboxes = bboxes.clone()
        x1y1 = bboxes[..., :2] - bboxes[..., 2:] / 2
        x2y2 = bboxes[..., :2] + bboxes[..., 2:] / 2

        x1y1x2y2 = torch.cat((x1y1, x2y2), dim=1)
        x1y1x2y2 = torch.clamp(x1y1x2y2, min=0, max=max_size)

        return x1y1x2y2
    
    @classmethod
    def compute_iou(cls, x, y):
        import pdb
        pdb.set_trace()
        x1 = torch.max(x[..., 0], y[..., 0])
        y1 = torch.max(x[..., 1], y[..., 1])
        x2 = torch.min(x[..., 2], y[..., 2])
        y2 = torch.min(x[..., 3], y[..., 3])

        intersect = torch.clamp((x2-x1), 0) * torch.clamp((y2-y1), 0)
        unions = abs((x[..., 2] - x[..., 0]) * (x[..., 3]-x[..., 1])) + abs(([..., 2] - y[..., 0]) * (y[..., 3]-y[..., 1])) - intersect
        intersect[intersect.gt(0)] = intersect[intersect.gt(0)] / unions[intersect.gt(0)]

        return intersect