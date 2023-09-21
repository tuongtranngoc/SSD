from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
from . import *
import numpy as np
from typing import Tuple, List

import torch
import torchvision


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
    def xyxy_to_xcycwh(cls, bboxes: torch.Tensor):
        bboxes = bboxes.clone()
        wh = bboxes[..., 2:] - bboxes[..., :2]
        xcyc = bboxes[..., 2:] - wh / 2.0
        xcyxwh = torch.cat((xcyc, wh), dim=1)
        xcyxwh = torch.clamp(xcyxwh, min=0, max=1.0)

        return xcyxwh
    
    @classmethod
    def pairwise_ious(cls, x:torch.Tensor, y:torch.Tensor):
        x1 = torch.max(x[:, None, 0], y[..., 0])
        y1 = torch.max(x[:, None, 1], y[..., 1])
        x2 = torch.min(x[:, None, 2], y[..., 2])
        y2 = torch.min(x[:, None, 3], y[..., 3])

        intersect = torch.clamp((x2-x1), 0) * torch.clamp((y2-y1), 0)
        unions = abs((x[:, None, 2] - x[:, None,  0]) * (x[:, None, 3] - x[:, None, 1])) + \
                 abs((y[..., 2] - y[..., 0]) * (y[..., 3] - y[..., 1])) - intersect
        intersect[intersect.gt(0)] = intersect[intersect.gt(0)] / unions[intersect.gt(0)]

        return intersect

    @classmethod
    def decode_ssd(cls, pred_bboxes: torch.Tensor, dfboxes: torch.Tensor):
        dfboxes = dfboxes.clone()
        dfboxes = dfboxes.to(pred_bboxes.device)
        pred_bboxes = pred_bboxes.clone()
        # transform offset into cxcywh
        xcyc = pred_bboxes[..., :2] * (dfboxes[..., 2:] * cfg.default_boxes.variances[1]) + dfboxes[..., :2]
        wh = torch.exp(pred_bboxes[..., 2:] * cfg.default_boxes.variances[0]) * dfboxes[..., 2:]
        xcycwh = torch.cat((xcyc, wh), dim=1)
        return xcycwh
    
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
    
    @classmethod
    def nms(cls, pred_bboxes, pred_confs, pred_classes, iou_thresh, conf_thresh):
        conf_mask = torch.where(pred_confs>=conf_thresh)[0]
        pred_bboxes = pred_bboxes[conf_mask]
        pred_confs = pred_confs[conf_mask]
        pred_classes = pred_classes[conf_mask]
    
        idxs = torchvision.ops.nms(pred_bboxes, pred_confs, iou_thresh)
        nms_bboxes = pred_bboxes[idxs]
        nms_confs = pred_confs[idxs]
        nms_classes = pred_classes[idxs]
    
        return nms_bboxes, nms_confs, nms_classes


class DataUtils:
    
    @classmethod
    def to_device(cls, data):
        if isinstance(data, torch.Tensor):
            return data.to(cfg.device)
        elif isinstance(data, Tuple) or isinstance(data, List):
            for i, d in enumerate(data):
                if isinstance(d, torch.Tensor):
                    data[i] = d.to(cfg.device)
                else:
                    Exception(f"{d} in {data} is not a tensor type")
            return data
        elif isinstance(data, torch.nn.Module):
            return data.to(cfg.device)
        else:
            Exception(f"{data} is not a/tuple/list of tensor type")

    @classmethod
    def to_numpy(cls, data):
        if isinstance(data, list):
            for i in range(len(data)):
                data[i] = cls.single_to_numpy(data[i])
            return data
        else:
            raise Exception(f"{data} is a type of {type(data)}, not list type")
    
    @classmethod
    def single_to_numpy(cls, data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise Exception(f"{data} is a type of {type(data)}, not numpy/tensor type")
    
    @classmethod
    def denormalize(cls, image):
        mean = np.array(cfg.voc_dataset.mean, dtype=np.float32)
        std = np.array(cfg.voc_dataset.std, dtype=np.float32)
        image *= (std * 255.)
        image += (mean * 255.)
        image = np.clip(image, 0, 255.)
        return image
        
    @classmethod
    def image_to_numpy(cls, image):
        if isinstance(image, torch.Tensor):
            if image.dim() > 3:
                image = image.squeeze()
            image = image.detach().cpu().numpy()
            image = image.transpose((1, 2, 0))
            image = cls.denormalize(image)
            image = np.ascontiguousarray(image, np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        elif isinstance(image, np.ndarray):
            image = cls.denormalize(image)
            image = np.ascontiguousarray(image, np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        else:
            raise Exception(f"{image} is a type of {type(image)}, not numpy/tensor type")

