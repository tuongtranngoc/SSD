from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import glob
import cv2
import os

from .augmentations import AlbumAug
from .base import BaseDataset
from .utils import Transform
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
    
    def __len__(self): return len(self.coco_dataset)

    def __getitem__(self, index):
        image_pth, labels = self.coco_dataset[index]
        
    
