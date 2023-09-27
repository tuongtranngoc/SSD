from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import albumentations as A

from .utils import *


class AlbumAug:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.ToGray(p=0.3),
            A.HorizontalFlip(p=0.3),
            A.Affine(p=0.3, rotate=15),
            A.BBoxSafeRandomCrop(p=0.3),
            A.Blur(p=0.3, blur_limit=5),
            A.RandomBrightnessContrast(p=0.3),
            A.MedianBlur(p=0.3, blur_limit=5),
            A.ShiftScaleRotate(p=0.3, rotate_limit=15),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.2))
    
    def __call__(self, image, bboxes, labels):
        transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
        transformed_image = transformed['image']
        transformed_bboxes = np.array(transformed['bboxes'], dtype=np.float32)
        transformed_labels = transformed['labels']
        if transformed_bboxes.shape[0] == 0:
            return image, bboxes, labels
        return transformed_image, transformed_bboxes, transformed_labels