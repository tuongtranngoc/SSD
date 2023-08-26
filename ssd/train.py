from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import *

class Trainer:
    def __init__(self) -> None:
        self.create_model()
        self.create_data_loader()

    def create_data_loader(self):
        self.train_dataset = COCODataset(cfg.dataset.train_label_path, cfg.dataset.train_img_path, cfg.training.is_augment)
        self.valid_dataset = COCODataset(cfg.dataset.val_label_path, cfg.dataset.val_img_path, cfg.valid.is_augment)
        self.train_loader = DataLoader(self.train_dataset, 
                                       batch_size=cfg.training.batch_size, 
                                       shuffle=cfg.training.shuffle,
                                       num_workers=cfg.training.num_workers,
                                       pin_memory=cfg.training.pin_memory)
        
    def create_model(self):
        self.model = SSDModel(arch_name=cfg.models.arch_name, pretrained=cfg.models.pretrained).to(cfg.device)
        self.loss_fn = SSDLoss()
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=cfg.training.lr, amsgrad=True)

    
    def train(self):
        for epoch in range(cfg.training.epochs):
            mt_box_loss = BatchMeter()
            mt_cls_loss = BatchMeter()

            for bz, (images, labels) in enumerate(self.train_loader):
                self.model.train()
                images = DataUtils.to_devices(images)
                labels = DataUtils.to_devices(labels)
                out = self.model(images)
 
                reg_loss, cls_loss = self.loss_fn(labels, out)
                total_loss = reg_loss + cls_loss

                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()

                mt_box_loss.update(reg_loss.item())
                mt_cls_loss.update(cls_loss.item())

                print(f"Epoch {epoch} Batch {bz+1}/{len(self.train_loader)}, reg_loss: {mt_box_loss.get_value(): .5f}, class_loss: {mt_cls_loss.get_value():.5f}", end="\r")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()