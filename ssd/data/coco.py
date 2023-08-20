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
        return image, bboxes, labels

    def matching_defaulboxes(self, bboxes, class_ids):
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        class_ids = torch.tensor(class_ids, dtype=torch.int32)
        bboxes = BoxUtils.normalize_box(bboxes)
        defaultboxes_dict = DefaultBoxesGenerator.build_default_boxes()
        defaultboxes = DefaultBoxesGenerator.merge_defaultboxes(defaultboxes_dict)
        defaultboxes = BoxUtils.xcycwh_to_xyxy(defaultboxes)

        # Matching default boxes to any ground truth box with jaccard overlap higher than a threshold (0.5)
        ious = BoxUtils.pairwise_ious(bboxes, defaultboxes)
        max_ious, max_idxs = ious.max(dim=0)

        # Indicator for matching the i-th default box to the j-th ground truth box of category p
        dfbox_idx_candicators = torch.where(max_ious > cfg.default_boxes.iou_thresh)[0]
        iou_candicators = max_ious[dfbox_idx_candicators]
        gt_idx_candicators = max_idxs[dfbox_idx_candicators]

        # Preprocess before computing loss during training the model
        bboxes_candicators = bboxes[gt_idx_candicators]
        class_ids_candicators = class_ids[gt_idx_candicators]
        dfbox_candicators = defaultboxes[dfbox_idx_candicators]
        
        bboxes_candicators = BoxUtils.xyxy_to_xcycwh(bboxes_candicators)
        dfbox_candicators = BoxUtils.xyxy_to_xcycwh(dfbox_candicators)

        return bboxes_candicators, dfbox_candicators, class_ids_candicators


    def __len__(self): return len(self.coco_dataset)

    def __getitem__(self, index):
        image_pth, lables = self.coco_dataset[index]
        class_ids, bboxes = lables[:, 0], lables[:, 1:]
        image, bboxes, class_ids = self.get_image_label(image_pth, bboxes, class_ids)
        self.matching_defaulboxes(bboxes, class_ids)

    
