from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from ssd.utils.losses import SSDLoss
from ssd.models.modules.ssd import SSDModel

from . import *

logger = Logger.get_logger("TRAINING")


class LitSSD(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = SSDModel()
        self.loss_func = SSDLoss()
        self.best_acc = 0.0
        self.acc_metrics = BatchMeter()
        self.train_reg_loss = BatchMeter()
        self.train_cls_loss = BatchMeter()
        self.automatic_optimization = False

    def on_train_start(self):
        pass

    def training_step(self, batch, batch_idx):
        images, labels = batch
        bz = images.size(0)
        outs = self.model(images)
        reg_loss, cls_loss = self.loss_func(labels, outs)
        total_loss = reg_loss + cls_loss

        self.train_reg_loss.update(reg_loss)
        self.train_cls_loss.update(cls_loss)

        self.log("reg_loss", self.train_reg_loss.get_value("mean"), prog_bar=True)
        self.log("cls_loss", self.train_cls_loss.get_value("mean"), prog_bar=True)

        optim = self.optimizers()
        optim.zero_grad()
        self.manual_backward(total_loss)
        optim.step()

    def validation_step(self, batch, batch_idx):
        images, labels, labels_len = batch
        bz = images.size(0)
        outs = self.model(images)
        return 
        
    def on_validation_epoch_end(self):
        current_acc = self.val_metrics.get_value("mean")
        self.log("Acc", current_acc, prog_bar=True)
        logger.info(f"Acc: {current_acc :.3f}")
        if current_acc > self.best_acc:
            self.best_acc = current_acc
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg['Optimizer']['lr'], amsgrad=True)
        return [optimizer]

    def on_train_end(self):
        logger.info(f"Loss: {self.train_metrics.get_value('mean')}")
        self.log(f"Best acc: {self.best_acc :.3f}")