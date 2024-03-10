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

logger = Logger.get_logger("VIDEO_PREDICTOR")


class VideoPredictor:
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

    def run(self, video_path):
        os.makedirs(cfg.debug['prediction'], exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        size = (300, 300)
        fps = int(cap.get(5))
        logger.info(f"Video size: {size}")
        if cap.isOpened() is False:
            logger.info("Error opening video file")
            exit()
        res = cv2.VideoWriter(os.path.join(cfg.debug['prediction'], \
                                        os.path.basename(video_path).replace('mp4', 'avi')), \
                                        fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=fps, frameSize=size)
        i = 0
        while True:
            ret, frame = cap.read()
            if ret is True:
                i+=1
                logger.info(f"Frame {i}")
                image = self.__transform(frame)
                image = image.to(cfg.device)
                image = image.unsqueeze(0)

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
                    image = Visualizer.draw_objects(image, pred_bboxes, confs, cates, 0.85, type_obj='PRED', unnormalize=True)
                    res.write(image)

            else:
                break
        cap.release()
        res.release()
        

if __name__ == "__main__":
    video_pred = VideoPredictor()
    video_pred.run('images/test.mp4')
