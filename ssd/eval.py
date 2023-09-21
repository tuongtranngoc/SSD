from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import torch
from . import *
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

logger = Logger.get_logger("EVALUATE")


class SSDEvaluate:
    def __init__(self, dataset, model) -> None:
        self.model = model
        self.dataset = dataset
        self.loss_fun = SSDLoss().to(cfg.device)
        self.dataloader = DataLoader(self.dataset,
                                    batch_size=cfg.valid.batch_size, 
                                    shuffle=cfg.valid.shuffle, 
                                    num_workers=cfg.valid.num_workers,
                                    pin_memory=cfg.valid.pin_memory)
        self.dfboxes = DefaultBoxesGenerator.default_boxes.to(cfg.device)

    def cal_mAP(self, map_mt, pred_bbox, pred_conf, pred_class, gt_bbox, gt_conf, gt_class):
        """Mean Average Precision (mAP)
        Reference: https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html
        """
        preds = [{"boxes": pred_bbox, "scores": pred_conf, "labels": pred_class}]
        
        target = [{"boxes": gt_bbox, "scores": gt_conf, "labels": gt_class}]

        map_mt.update(preds, target)

    def evaluate(self):
        metrics = {
            "eval_reg_loss": BatchMeter(),
            "eval_cls_loss": BatchMeter(),
            "eval_map": BatchMeter(),
            "eval_map_50": BatchMeter(),
            "eval_map_75": BatchMeter()
        }

        map_mt = MeanAveragePrecision(class_metrics=True)
        self.model.eval()
        
        for i, (images, target_dfboxes, idxs) in enumerate(self.dataloader):
            with torch.no_grad():
                tensor_images = DataUtils.to_device(images)
                target_dfboxes = DataUtils.to_device(target_dfboxes)
                out = self.model(tensor_images)
                reg_loss, cls_loss = self.loss_fun(target_dfboxes, out)
                
                metrics["eval_reg_loss"].update(reg_loss)
                metrics["eval_cls_loss"].update(cls_loss)
                bpred_bboxes, bpred_confs = out
                
                for j, idx in enumerate(idxs):
                    img_path, targets = self.dataset.voc_dataset[idx]
                    target_labels, target_bboxes = targets[:, 0], targets[:, 1:]
                    target_confs = np.ones_like(target_labels, dtype=np.float32)
                    pred_bboxes, pred_confs = bpred_bboxes[j], bpred_confs[j]

                    # Normalize bboxes and to tensor
                    image, target_bboxes, target_labels = self.dataset.get_image_label(img_path, target_bboxes, target_labels, False)
                    target_bboxes = torch.tensor(target_bboxes, dtype=torch.float32, device=cfg.device)
                    target_bboxes = BoxUtils.normalize_box(target_bboxes)
                    target_labels = torch.tensor(target_labels, dtype=torch.long, device=cfg.device)
                    target_confs = torch.tensor(target_confs, dtype=torch.float32, device=cfg.device)

                    pred_confs = torch.softmax(pred_confs, dim=-1)
                    pred_confs, pred_cates = pred_confs.max(dim=-1)

                    # Filter negative predictions
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

                    # Calculate mAP
                    self.cal_mAP(map_mt, pred_bboxes, pred_confs ,pred_cates, target_bboxes, target_confs, target_labels)
        
        # Compute mAP
        mAP = map_mt.compute()
        metrics['eval_map'].update(mAP['map'])
        metrics['eval_map_50'].update(mAP['map_50'])
        metrics['eval_map_75'].update(mAP['map_75'])
        
        logger.info(f'reg_loss: {metrics["eval_reg_loss"].get_value("mean"): .3f}, cls_loss: {metrics["eval_cls_loss"].get_value("mean"): .3f}')
        logger.info(f'mAP: {metrics["eval_map"].get_value("mean"): .3f}, mAP_50: {metrics["eval_map_50"].get_value("mean"): .3f}, mAP_75: {metrics["eval_map_75"].get_value("mean"): .3f}')
        
        return metrics
    
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_type', type=str, default='last.pt', help='Types of the model weight: best.pt/last.pt')
    parser.add_argument('--model_type', type=str, default='vgg16', help='Model selection: vgg16, vgg16-bn, resnet34, resnet50')
    parser.add_argument('--batch_size', type=int, default=cfg.valid.batch_size, help='Batch size of evaluation')
    parser.add_argument('--conf_thresh', type=float, default=cfg.debug.conf_thresh, help='Confidence threshold for evaluation')
    parser.add_argument('--iou_thresh', type=float, default=cfg.debug.iou_thresh, help='Iou threshold for evaluation')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    dataset = VOCDataset(cfg.voc_dataset.anno_path, cfg.voc_dataset.image_path, cfg.voc_dataset.val_txt_path, cfg.valid.is_augment)
    model = SSDModel(pretrained=False).to(cfg.device)
    ckpt_path = os.path.join(cfg.debug.ckpt_dirpath, args.model_type, args.weight_type)
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    eval = SSDEvaluate(dataset, model)
    eval.evaluate()