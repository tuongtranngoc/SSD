from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import glob
import cv2
import os

from .base import BaseDataset

class COCODataset(BaseDataset):
    def __init__(self, label_path, image_path) -> None:
        super().__init__(label_path, image_path)
        