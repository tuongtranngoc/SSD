from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm

from . import *
from .predict import Predictor

logger = Logger.get_logger("VIDEO_PREDICTOR")


class VideoPredictor:
    def __init__(self) -> None:
        self.predictor = Predictor()

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        size = (int(cap.get(3)), int(cap.get(4)))
        if cap.isOpened() is False:
            logger.info("Error opening video file")
            exit()
        res = cv2.VideoWriter(os.path.join(cfg.debug['prediction'], \
                                        os.path.basename(video_path)), \
                                        fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=15, frameSize=size)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret is True:
                image = frame.copy()
                image = self.__transform(image)
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
                    image = Visualizer.draw_objects(image, pred_bboxes, confs, cates, cfg.debug.conf_thresh, type_obj='PRED', unnormalize=True)
                    res.write(image)
            else:
                break
        cap.release()
        res.release()
        

if __name__ == "__main__":
    video_pred = VideoPredictor()
    video_pred.run('')
