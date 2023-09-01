import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from . import *


class Transform:
    mean = cfg.dataset.mean
    std = cfg.dataset.std
    image_size = cfg.models.image_size
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(),
        ToTensorV2()],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    @classmethod
    def transfrom(cls, image, bboxes, labels):
        transformed = cls.transform(image=image, bboxes=bboxes, labels=labels)
        transformed_image = transformed['image'] 
        transformed_bboxes = np.array(transformed['bboxes'], dtype=np.float32)
        transformed_labels = transformed['labels']
        return transformed_image, transformed_bboxes, transformed_labels

    @classmethod
    def denormalize(cls, image):
        mean = np.array(cls.mean, dtype=np.float32)
        std = np.array(cls.std, dtype=np.float32)
        image *= (std * 255.)
        image += (mean * 255.)
        image = np.clip(image, 0, 255.)
        return image