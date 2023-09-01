from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import random
from . import *
import numpy as np
from collections import defaultdict


class Visualization:

    h, w = cfg.models.image_size, cfg.models.image_size
    thickness = 1
    lineType = cv2.LINE_AA

    @classmethod
    def name_id_classes(cls):
        class_names = defaultdict()
        with open(cfg.dataset.coco_classes, 'r') as f:
            for i, l in enumerate(f.readlines()):
                class_names[l.strip()] = i
        f.close()
        return class_names

    @classmethod
    def colors(cls):
        colors = {
            k: tuple([random.randint(0, 255) for _ in range(3)])
            for k in cls.name_id_classes.keys()
        }
        colors['groundtruth'] = (255, 0, 0)
        colors['background'] = (128, 128, 128)
        return colors
    
    @classmethod
    def unnormalize_box(cls, bboxes:np.ndarray):
        bboxes = bboxes.copy()
        bboxes[..., [0, 2]] *= cls.w
        bboxes[..., [1, 3]] *= cls.h
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]].clip(min=0.0, max=cls.w)
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]].clip(min=0.0, max=cls.h)
        return bboxes

    @classmethod
    def draw_object(cls, image, bbox, conf, label,  type_obj=None):
        bbox = cls.unnormalize_box(bbox)
        if type_obj == 'GT':
            color = cls.colors['groundtruth']
            text = label
        elif type_obj == 'PRED':
            color = cls.colors[label]
            text = '_'.join([label, str(round(conf, 3))])
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