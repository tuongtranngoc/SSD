from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch.nn.functional as F
import numpy as np
import torch
import glob
import cv2
import os

from .default_boxes import DefaultBoxesGenerator
from .augmentations import AlbumAug
from .base import BaseDataset
from .utils import Transformation
from . import BoxUtils
from . import cfg


class VOCDataset(BaseDataset):
    def __init__(self, label_path, image_path, txt_path, is_augment=False) -> None:
        super().__init__(label_path, image_path, txt_path)
        self.aug = AlbumAug()
        self.is_augment = is_augment
        self.voc_dataset = self.load_voc_dataset()
        self.__tranform = Transformation()

    def get_image_label(self, image_pth, bboxes, labels, is_aug):
        image = cv2.imread(image_pth)
        image = image[..., ::-1]
        if is_aug:
            image, bboxes, labels = self.aug(image, bboxes, labels)
        image, bboxes, labels = self.__tranform.transform(image, bboxes, labels)
        return image, bboxes, labels

    def matching_defaulboxes(self, bboxes, class_ids):
        bboxes = torch.tensor(bboxes.copy(), dtype=torch.float32)
        class_ids = torch.tensor(class_ids, dtype=torch.long)

        bboxes = BoxUtils.normalize_box(bboxes)
        defaultboxes_dict = DefaultBoxesGenerator.build_default_boxes()
        defaultboxes = DefaultBoxesGenerator.merge_defaultboxes(defaultboxes_dict)
        defaultboxes = BoxUtils.xcycwh_to_xyxy(defaultboxes)
        
        # Create mask for matched defaultboxes
        dfboxes_mask = torch.zeros_like(defaultboxes, dtype=torch.float32)
        dflabels_mask = torch.zeros(defaultboxes.size(0), dtype=torch.long)
        
        # Matching default boxes to any ground truth box with jaccard overlap higher than a threshold (0.5)
        ious = BoxUtils.pairwise_ious(bboxes, defaultboxes)
        max_ious, max_idxs = ious.max(dim=0)

        # Indicator for matching the i-th default box to the j-th ground truth box of category p
        dfbox_idx_pos = torch.where(max_ious > cfg.default_boxes.iou_thresh)[0]
        iou_pos = max_ious[dfbox_idx_pos]
        gt_idx_pos = max_idxs[dfbox_idx_pos]
        
        # Preprocess before computing loss during training the model
        bboxes_pos = bboxes[gt_idx_pos]
        dflabels_mask[dfbox_idx_pos] = class_ids[gt_idx_pos]
        dfbox_pos = defaultboxes[dfbox_idx_pos]
        
        # Convert xyxy to xcyxwh and simplify targets
        bboxes_pos = BoxUtils.xyxy_to_xcycwh(bboxes_pos)
        dfbox_pos = BoxUtils.xyxy_to_xcycwh(dfbox_pos)
        
        dfbox_pos = self.encode_ssd(bboxes_pos, dfbox_pos)
        dfboxes_mask[dfbox_idx_pos] = dfbox_pos

        return dfboxes_mask, dflabels_mask
        
    def encode_ssd(self, gt_bboxes, df_bboxes):
        # Simplify the location of default boxes
        g_cxcy = (gt_bboxes[..., :2] - df_bboxes[..., :2]) / df_bboxes[..., 2:]
        g_wh = torch.log(gt_bboxes[..., 2:] / df_bboxes[..., 2:])
        gm = torch.cat((g_cxcy, g_wh), dim=1)
        return gm
    
    def __len__(self): return len(self.voc_dataset)
    
    def __getitem__(self, index):
        image_pth, labels = self.voc_dataset[index]
        class_ids, bboxes = labels[:, 0], labels[:, 1:]
        image, bboxes, class_ids = self.get_image_label(image_pth, bboxes, class_ids, self.is_augment)
        targets_dfboxes = self.matching_defaulboxes(bboxes, class_ids)
        return image, targets_dfboxes, index