from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import random
import numpy as np
from collections import defaultdict

import torch
from . import *
import torch.nn.functional as F


class COCOAnnotation:
    """COCO Annotation: color, class ids
    """
    def __init__(self) -> None:
        self.class_names = defaultdict()
        with open(cfg.dataset.coco_classes, 'r') as f:
            for i, l in enumerate(f.readlines()):
                self.class_names[l.strip()] = i
        f.close()

    def class2color(self, cls_name):
        colors = {k: tuple([random.randint(0, 255) for _ in range(3)])
                    for k in self.class_names.keys()}
        colors['groundtruth'] = (255, 0, 0)
        colors['background'] = (128, 128, 128)
        return colors[cls_name]

    def id2class(self, cls_id):
        ids = {v:k for k, v in self.class_names.items()}
        return ids[cls_id]


class Visualizer:
    """ Visualize debug images from training and valid
    """
    thickness = 1
    lineType = cv2.LINE_AA
    coco_ano = COCOAnnotation()
    h, w = cfg.models.image_size, cfg.models.image_size
    dfboxes = DefaultBoxesGenerator.build_default_boxes()
    dfboxes = DefaultBoxesGenerator.merge_defaultboxes(dfboxes).to(cfg.device)
    
    @classmethod
    def unnormalize_box(cls, bboxes:np.ndarray):
        bboxes = bboxes.copy()
        bboxes[..., [0, 2]] *= cls.w
        bboxes[..., [1, 3]] *= cls.h
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]].clip(min=0.0, max=cls.w)
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]].clip(min=0.0, max=cls.h)
        return bboxes

    @classmethod
    def draw_objects(cls, image, bboxes, confs, labels, conf_thresh, type_obj=None):
        for bbox, conf, label in zip(bboxes, confs, labels):
            if conf >= conf_thresh:
                image = cls.single_draw_object(image, bbox, conf, label, type_obj)
        return image

    @classmethod
    def single_draw_object(cls, image, bbox, conf, label,  type_obj=None):
        bbox = cls.unnormalize_box(bbox)
        label = cls.coco_ano.id2class(label)
        if type_obj == 'GT':
            color = cls.coco_ano.class2color('groundtruth')
            text = label
        elif type_obj == 'PRED':
            color = cls.coco_ano.class2color(label)
            text = '-'.join([label, str(round(conf, 3))])
        else:
            Exception(f"Not have type_obj is None")

        cv2.rectangle(image,
                    (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                    color=color,
                    thickness=cls.thickness,
                    lineType=cls.lineType)
        
        cv2.putText(image, text,
                    (int(bbox[0]), int(bbox[1] + 0.025*cls.w)),
                    fontFace=0,
                    fontScale=cls.thickness/3,
                    color=color,
                    thickness=cls.thickness,
                    lineType=cls.lineType)
        
        return image

    @classmethod
    def debug_output(cls, dataset, idxs, model, type_fit, debug_dir, apply_nms=True):
        os.makedirs(os.path.join(debug_dir, type_fit), exist_ok=True)
        model.eval()
        images = []
        bboxes = []
        labels = []
        for idx in idxs:
            image, target = dataset[idx]
            images.append(image)
            bboxes.append(target[0])
            labels.append(target[1])
        
        images = torch.stack(images, dim=0).to(cfg.device)
        bboxes = torch.stack(bboxes, dim=0).to(cfg.device)
        labels = torch.stack(labels, dim=0).to(cfg.device)

        bpred_bboxes, bpred_confs = model(images) # (N, boxes, 4) & (N, boxes, num classes)

        for i in range(images.size(0)):
            target_bboxes, target_labels = bboxes[i], labels[i]
            target_confs = torch.ones_like(target_labels, dtype=torch.float32, device=cfg.device)
            pos_mask = target_labels > 0
            target_bboxes = target_bboxes[pos_mask]
            target_labels = target_labels[pos_mask]
            target_confs = target_confs[pos_mask]
            dfboxes = cls.dfboxes[pos_mask]
            
            target_bboxes = BoxUtils.decode_ssd(target_bboxes, dfboxes)
            target_bboxes = BoxUtils.xcycwh_to_xyxy(target_bboxes)

            pred_bboxes, pred_confs = bpred_bboxes[i], bpred_confs[i]
            pred_bboxes = BoxUtils.decode_ssd(pred_bboxes, cls.dfboxes)
            pred_bboxes = BoxUtils.xcycwh_to_xyxy(pred_bboxes)
            
            pred_confs = F.softmax(pred_confs, dim=-1)
            confs, cates = pred_confs.max(dim=-1)
    
            if apply_nms:
                pred_bboxes, confs, cates = BoxUtils.nms(pred_bboxes, confs, cates, cfg.debug.iou_thresh, cfg.debug.conf_thresh)

            target_bboxes, target_confs, target_labels = DataUtils.to_numpy(target_bboxes, target_confs, target_labels)
            pred_bboxes, confs, cates = DataUtils.to_numpy(pred_bboxes, confs, cates)

            image = DataUtils.image_to_numpy(images[i])

            image = cls.draw_objects(image, target_bboxes, target_confs, target_labels, cfg.debug.conf_thresh, type_obj='GT')
            image = cls.draw_objects(image, pred_bboxes, confs, cates, cfg.debug.conf_thresh, type_obj='PRED')

            cv2.imwrite(os.path.join(debug_dir, type_fit, f'{i}.png'), image)