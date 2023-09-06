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


class AnnotationTool:
    """Annotation: color, class ids
    """
    def __init__(self) -> None:
        self.class_names = defaultdict()
        with open(cfg.voc_dataset.classes, 'r') as f:
            for i, l in enumerate(f.readlines()):
                self.class_names[l.strip()] = i
        f.close()
        self.colors = {k: tuple([random.randint(0, 255) for _ in range(3)])
                    for k in self.class_names.keys()}

    def class2color(self, cls_name):
        self.colors['groundtruth'] = (0, 0, 255)
        self.colors['background'] = (128, 128, 128)
        return self.colors[cls_name]

    def id2class(self, cls_id):
        ids = {v:k for k, v in self.class_names.items()}
        return ids[cls_id]


class Visualizer:
    """ Visualize debug images from training and valid
    """
    thickness = 1
    lineType = cv2.LINE_AA
    cvt_ano = AnnotationTool()
    h, w = cfg.models.image_size, cfg.models.image_size
    dfboxes = DefaultBoxesGenerator.default_boxes.to(cfg.device)

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
        if label == 0: return image
        bbox = cls.unnormalize_box(bbox)
        label = cls.cvt_ano.id2class(label)
        if type_obj == 'GT':
            color = cls.cvt_ano.class2color('groundtruth')
            text = label
        elif type_obj == 'PRED':
            color = cls.cvt_ano.class2color(label)
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
        for i, idx in enumerate(idxs):
            img_path, targets = dataset.voc_dataset[idx]
            target_labels, target_bboxes = targets[..., 0], targets[..., 1:]
            target_confs = np.ones_like(target_labels, dtype=np.float32)

            # Normalize bboxes
            image, target_bboxes, target_labels = dataset.get_image_label(img_path, target_bboxes, target_labels, False)
            target_bboxes =  torch.tensor(target_bboxes, dtype=torch.float32, device=cfg.device)
            target_bboxes = BoxUtils.normalize_box(target_bboxes)
            target_confs =  torch.tensor(target_confs, dtype=torch.float32, device=cfg.device)
            target_labels = torch.tensor(target_labels, dtype=torch.long, device=cfg.device)
            # Decode bboxes
            pred_bboxes, pred_confs = model(image.to(cfg.device).unsqueeze(0))
            pred_bboxes, pred_confs = pred_bboxes.squeeze(0), pred_confs.squeeze(0)
            pred_bboxes = BoxUtils.decode_ssd(pred_bboxes, cls.dfboxes)
            pred_bboxes = BoxUtils.xcycwh_to_xyxy(pred_bboxes)
            
            pred_confs = torch.softmax(pred_confs, dim=-1)
            confs, cates = pred_confs.max(dim=-1)
            # Filter negative predictions
            pred_pos_mask = cates > 0
            pred_bboxes = pred_bboxes[pred_pos_mask]
            confs = confs[pred_pos_mask]
            cates = cates[pred_pos_mask]
            # Apply non-max suppression
            if apply_nms:
                pred_bboxes, confs, cates = BoxUtils.nms(pred_bboxes, confs, cates, cfg.debug.iou_thresh, cfg.debug.conf_thresh)
            # Tensor to numpy
            target_bboxes, target_confs, target_labels = DataUtils.to_numpy(target_bboxes, target_confs, target_labels)
            pred_bboxes, confs, cates = DataUtils.to_numpy(pred_bboxes, confs, cates)
            image = DataUtils.image_to_numpy(image)
            # Draw debug images
            image = cls.draw_objects(image, target_bboxes, target_confs, target_labels, cfg.debug.conf_thresh, type_obj='GT')
            image = cls.draw_objects(image, pred_bboxes, confs, cates, cfg.debug.conf_thresh, type_obj='PRED')

            cv2.imwrite(os.path.join(debug_dir, type_fit, f'{i}.png'), image)