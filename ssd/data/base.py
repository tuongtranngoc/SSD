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

from . import cfg
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

        coco_classes = {
            0 : 'background'
        }
        for cat in data['categories']:
            coco_classes[cat['id']] = cat['name']
        
        list_ids = coco_classes.keys()

        new_class_names = defaultdict()
        with open(cfg.dataset.coco_classes, 'r') as f:
            for i, l in enumerate(f.readlines()):
                new_class_names[l.strip()] = i
        f.close()

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
                if ann['iscrowd'] or ann['category_id'] not in list_ids: continue
                bbox = ann['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                if int(bbox[2]) <= int(bbox[0]) or int(bbox[3]) <= int(bbox[1]):
                    continue
                cate = ann['category_id']
                bbox_info = [new_class_names[coco_classes[cate]]] + bbox
                bboxes.append(bbox_info)
            dataset.append([img_path, np.array(bboxes)])
        
        return dataset




