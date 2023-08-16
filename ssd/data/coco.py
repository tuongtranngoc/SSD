from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import glob
import cv2
import os

from .default_boxes import DefaultBoxesGenerator
from .augmentations import AlbumAug
from .base import BaseDataset
from .utils import Transform
from . import BoxUtils
from . import cfg


class COCODataset(BaseDataset):
    def __init__(self, label_path, image_path, is_augment=False) -> None:
        super().__init__(label_path, image_path)
        self.aug = AlbumAug()
        self.is_augment = is_augment
        self.transform = Transform(cfg.models.image_size)
        self.coco_dataset = self.load_coco_dataset()

    def get_image_label(self, image_pth, bboxes, labels):
        image = cv2.imread(image_pth)
        if self.is_augment:
            image, bboxes, labels = self.aug(image, bboxes, labels)
        image = image[..., ::-1]
        image, bboxes, labels = self.transform(image, bboxes, labels)
        return image, image, bboxes, labels

    def match_defaulboxes(self, id_cls, bboxes):
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        bboxes = BoxUtils.normalize_box(bboxes)
        defaultboxes_dict = DefaultBoxesGenerator.build_default_boxes()
        defaultboxes = DefaultBoxesGenerator.merge_defaultboxes(defaultboxes_dict)
        defaultboxes = BoxUtils.xcycwh_to_xyxy(defaultboxes)
        ious = BoxUtils.compute_iou(bboxes, defaultboxes)
        matched_dfboxes = defaultboxes[ious > cfg.default_boxes.iou_thresh]
        return matched_dfboxes
        
    
    def __len__(self): return len(self.coco_dataset)

    def __getitem__(self, index):
        image_pth, lables = self.coco_dataset[index]
        cls_ids, bboxes = lables[:, 0], lables[:, 1:]
        image, bboxes, cls_ids = self.get_image_label(image_pth, bboxes, lables)
        self.match_defaulboxes(cls_ids, bboxes)

    
