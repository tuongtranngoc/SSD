from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import numpy as np
import albumentations as A

from .utils import *


class AlbumAug:
    def __init__(self) -> None:
        self.random_iou_crop = RandomIouCrop()
        self.__transform = A.Compose([
            A.BBoxSafeRandomCrop(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Affine(p=0.3, rotate=15),
            A.ShiftScaleRotate(p=0.2, rotate_limit=15),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20),
            # A.ToGray(p=0.01),
            A.Blur(p=0.01, blur_limit=5),
            A.MedianBlur(p=0.01, blur_limit=5),
            A.RandomBrightnessContrast(p=0.3),
            ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.1))
    
    def __call__(self, image, bboxes, labels):
        # image, bboxes, labels = self.random_iou_crop(image, bboxes, labels)
        transformed = self.__transform(image=image, bboxes=bboxes, labels=labels)
        transformed_image = transformed['image']
        transformed_bboxes = np.array(transformed['bboxes'])
        if transformed_bboxes.shape[0] == 0:
            return image, bboxes, labels
        transformed_labels = np.array(transformed['labels'])
        return transformed_image, transformed_bboxes, transformed_labels


class RandomIouCrop:
    def __init__(self, p: float = 0.5, min_scale: float = 0.5,
                 max_scale: float = 1.0, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2.0,
                 sampler_options = None, trials: int = 30):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials
        self.p = p

    def __call__(self, img: np.ndarray, bboxes: np.ndarray, labels: np.ndarray):
        if np.random.uniform(0, 1) < self.p:
            return img, bboxes, labels
        img = img.copy()
        orig_w, orig_h = img.shape[:2]
        while True:
            idx = int(np.random.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:
                return img, bboxes, labels
            
            for _ in range(self.trials):
                r = self.min_scale + (self.max_scale-self.min_scale) * np.random.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue
                r = np.random.rand(2)
                left = int((orig_w-new_w)*r[0])
                top = int((orig_h-new_h)*r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                cx = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
                cy = 0.5 * (bboxes[:, 1] + bboxes[:, 3])

                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                boxes = bboxes[is_within_crop_area]
                ious = box_iou(boxes, np.array([left, top, right, bottom], dtype=boxes.dtype))
                if ious.max() < min_jaccard_overlap:
                    continue

                bboxes = boxes
                labels = labels[is_within_crop_area]
                bboxes[:, 0::2] -= left
                bboxes[:, 1::2] -= top
                bboxes[:, 0::2] = bboxes[:, 0::2].clip(min=1, max=new_w)
                bboxes[:, 1::2] = bboxes[:, 1::2].clip(min=1, max=new_h)
                img = img[top:top+new_h, left:left+new_w]
                return img, bboxes, labels
