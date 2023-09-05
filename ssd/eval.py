from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
from . import *
import argparse
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

logger = Logger.get_logger("EVALUATE")


class SSDEvaluate:
    def __init__(self, dataset, model) -> None:
        self.model = model
        self.loss_fun = SSDLoss().to(cfg.device)
        self.dataloader = DataLoader(dataset,
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

        for i, (images, labels) in enumerate(self.dataloader):
            with torch.no_grad():
                images = DataUtils.to_device(images)
                labels = DataUtils.to_device(labels)
                out = self.model(images)
                bz = images.size(0)
                reg_loss, cls_loss = self.loss_fun(labels, out)

                metrics["eval_reg_loss"].update(reg_loss)
                metrics["eval_cls_loss"].update(cls_loss)

                btarget_bboxes, btarget_labels = labels
                bpred_bboxes, bpred_confs = out

                for j in range(bz):
                    target_confs = torch.ones_like(target_labels, dtype=torch.float32, device=cfg.device)
                    target_bboxes, target_labels = btarget_bboxes[j], btarget_labels[j]
                    pred_bboxes, pred_confs = bpred_bboxes[j], bpred_confs[j]
                    # Filter negative targets
                    target_pos_mask = target_labels > 0
                    target_bboxes = target_bboxes[target_pos_mask]
                    target_labels = target_labels[target_pos_mask]
                    target_confs = target_confs[target_pos_mask]
                    dfboxes = self.dfboxes[target_pos_mask]

                    # Decode target bboxes
                    target_bboxes = BoxUtils.decode_ssd(target_bboxes, dfboxes)
                    target_bboxes = BoxUtils.xcycwh_to_xyxy(target_bboxes)

                    # Filter negative predictions
                    pred_confs = torch.softmax(pred_confs, dim=-1)
                    pred_confs, pred_cates = pred_confs.max(dim=-1)
                    pred_pos_mask = pred_cates > 0
                    pred_bboxes = pred_bboxes[pred_pos_mask]
                    pred_confs = pred_confs[pred_pos_mask]
                    pred_cates = pred_cates[pred_cates]

                    # Decode predicted bboxes
                    pred_bboxes = BoxUtils.decode_ssd(pred_bboxes, self.dfboxes)
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
    parser.add_argument('--weight_type', type=str, default='best.pt', help='Types of the model weight: best.pt/last.pt')
    parser.add_argument('--model_type', type=str, default='vgg16', help='Model selection: vgg16, vgg16-bn, resnet34, resnet50')
    parser.add_argument('--batch_size', type=int, default=cfg.valid.batch_size, help='Batch size of evaluation')
    parser.add_argument('--conf_thresh', type=float, default=cfg.debug.conf_thresh, help='Confidence threshold for evaluation')
    parser.add_argument('--iou_thresh', type=float, default=cfg.debug.iou_thresh, help='Iou threshold for evaluation')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    pass