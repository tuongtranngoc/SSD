from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from . import *


class Predictor:
    def __init__(self) -> None:
        self.model = SSDModel(cfg.models.arch_name).to(cfg.device)
        self.model = self.load_ckpt(self.model, os.path.join(cfg.debug.ckpt_dirpath, cfg.models.arch_name, 'best.pt'))
        self.transform = A.Compose([
            A.Resize(cfg.models.image_size, cfg.models.image_size),
            A.Normalize(),
            ToTensorV2()
        ])
        self.dfboxes = DefaultBoxesGenerator.df_bboxes.to(cfg.device)

    def __transform(self, image):
        image = image[..., ::-1]
        image = self.transform(image=image)
        return image['image']
    
    def load_ckpt(self, model, ckpt_pth):
        if os.path.exists(ckpt_pth):
            ckpt = torch.load(ckpt_pth, map_location=cfg.device)
            model.load_state_dict(ckpt['model'])
            return model
        else:
            raise Exception(f'Path to the model {ckpt_pth} not exist')

    def predict(self, image_path):
        image = cv2.imread(image_path)
        image = self.__transform(image)
        image = image.to(cfg.device)
        image = image.unsqueeze(0)
        self.model.eval()

        with torch.no_grad():
            pred_bboxes, pred_confs = self.model(image)
            pred_bboxes, pred_confs = pred_bboxes.squeeze(), pred_confs.squeeze()
            pred_confs = torch.softmax(pred_confs, dim=-1)
            pred_confs, pred_cates = pred_confs.max(dim=-1)
            pred_pos_mask = pred_cates > 0

            pred_bboxes = pred_bboxes[pred_pos_mask]
            pred_confs = pred_confs[pred_pos_mask]
            pred_cates = pred_cates[pred_pos_mask]
            pred_dfboxes = self.dfboxes[pred_pos_mask]
            
            # Decode predicted bboxes
            pred_bboxes = BoxUtils.decode_ssd(pred_bboxes, pred_dfboxes)
            pred_bboxes = BoxUtils.xcycwh_to_xyxy(pred_bboxes)
            
            # Apply non-max suppression
            pred_bboxes, pred_confs, pred_cates = BoxUtils.nms(pred_bboxes, pred_confs, pred_cates, cfg.debug.iou_thresh, cfg.debug.conf_thresh)
            # Tensor to numpy
            pred_bboxes, confs, cates = DataUtils.to_numpy([pred_bboxes, pred_confs, pred_cates])
            image = DataUtils.image_to_numpy(image)
            # Visualize debug images
            image = Visualizer.draw_objects(image, pred_bboxes, confs, cates, cfg.debug.conf_thresh, type_obj='PRED', unnormalize=True)
            os.makedirs(cfg.debug.prediction, exist_ok=True)
            cv2.imwrite(f'{os.path.join(cfg.debug.prediction, os.path.basename(image_path))}', image)

if __name__ == "__main__":
    predictor = Predictor()
    IMAGE_ID = "dataset/VOC/images_id/test2007.txt"
    IMAGE_PTH = "dataset/VOC/images/test2007"
    with open(IMAGE_ID, 'r') as f_id:
        list_img_ids = f_id.readlines()
        for img_id in tqdm(list_img_ids):
            img_id = img_id.strip()
            image_path = os.path.join(IMAGE_PTH, img_id + '.jpg')
            result = predictor.predict(image_path)