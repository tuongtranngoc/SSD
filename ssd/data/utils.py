import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from . import *


class Transformation:
    def __init__(self) -> None:
        image_size = cfg.models.image_size
        self.transformation = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(always_apply=True),
            ToTensorV2()],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    def transform(self, image, bboxes, labels):
        transformed = self.transformation(image=image, bboxes=bboxes, labels=labels)
        transformed_image = transformed['image'] 
        transformed_bboxes = np.array(transformed['bboxes'], dtype=np.float32)
        transformed_labels = np.array(transformed['labels'])
        return transformed_image, transformed_bboxes, transformed_labels