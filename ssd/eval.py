from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from . import *
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision

logger = Logger.get_logger("EVALUATE")


class CocoEvaluate:
    def __init__(self, dataset, model) -> None:
        self.model = model
        self.loss_fun = SSDLoss().to(cfg.device)
        self.dataloader = DataLoader(dataset,
                                    batch_size=cfg.valid.batch_size, 
                                    shuffle=cfg.valid.shuffle, 
                                    num_workers=cfg.valid.num_workers,
                                    pin_memory=cfg.valid.pin_memory)
        
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
                reg_loss, cls_loss = self.loss_fun(labels.clone(), out)

                metrics["eval_reg_loss"].update(reg_loss)
                metrics["eval_cls_loss"].update(cls_loss)

        logger.info(f'reg_loss: {metrics["eval_reg_loss"].get_value("mean"): .3f}, cls_loss: {metrics["eval_cls_loss"].get_value("mean"): .3f}')

        return metrics