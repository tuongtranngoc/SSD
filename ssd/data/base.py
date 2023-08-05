from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import os
import cv2
import glob
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, label_path, image_path) -> None:
        self.label_path = label_path
        self.image_path = image_path

    def get_image(self, img_pth):
        im = cv2.imread(img_pth)
        im = im[..., ::-1]
        return im

    def get_labels(self):
        with open(self.label_path, 'r') as f_label:
            data = json.load(f_label)
        f_label.close()
        return data

    def load_coco_dataset(self):
        dataset = []
        data = self.get_labels()

        images = {
            x['id']: x
            for x in data['images']
        }
        annotations = data['annotations']
        anns_imgid = defaultdict(list)

        for anns in annotations:
            anns_imgid[anns['image_id']].append(anns)

        for img_id, anns in tqdm(anns_imgid.items(), desc="Parsing coco data"):
            image = images[img_id]
            fname, w, h = image['file_name'], image['width'], image['height']
            img_path = os.path.join(self.image_path, fname)
            if not os.path.exists(img_path):
                continue
            
            bboxes = []
            for ann in anns:
                if not ann['iscrowd']: continue
                bbox = ann['bbox']
                cate = ann['category_id'] - 1
                bbox_info = bbox + [cate]
                bboxes.append(bbox_info)

            dataset.append([img_path, np.array(bboxes)])

        return dataset




