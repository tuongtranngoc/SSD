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
    

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def box_iou(box_a, box_b)->np.ndarray:
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])) 
    area_b = ((box_b[2]-box_b[0]) * (box_b[3]-box_b[1]))  
    union = area_a + area_b - inter
    return inter / union 