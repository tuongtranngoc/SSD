from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from . import COCODataset
from . import cfg


class COCODataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.train_dataloader_params = {
            'batch_size': cfg.dataset.batch_size,
            'shuffle': cfg.dataset.shuffle,
            'num_workers': cfg.dataset.num_workers,
            'pin_memory': cfg.dataset.pin_memory
        }

        self.val_dataloader_params = {
            'batch_size': cfg.dataset.batch_size,
            'shuffle': cfg.dataset.shuffle,
            'num_workers': cfg.dataset.num_workers,
            'pin_memory': cfg.dataset.pin_memory
        }

        self.train_label_path = cfg.dataset.train_label_path
        self.train_img_path = cfg.dataset.train_img_path
        self.train_is_augment = cfg.training.is_augment

        self.val_label_path = cfg.dataset.val_label_path
        self.val_img_path = cfg.dataset.val_img_path
        self.val_is_augment = cfg.valid.is_augment
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_coco = COCODataset(self.train_label_path, self.train_img_path, self.train_is_augment)
            self.val_coco = COCODataset(self.val_label_path, self.val_img_path, self.val_is_augment)
    
    def train_dataloader(self):
        return DataLoader(self.train_coco, **self.train_loader_params)
    
    def val_dataloader(self):
        return DataLoader(self.val_coco, **self.val_dataloader_params)
    