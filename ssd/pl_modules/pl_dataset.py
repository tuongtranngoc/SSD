from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from . import *


class VocModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super(VocModule, self).__init__()
    
    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = VOCDataset(cfg.voc_dataset.anno_path, 
                                            cfg.voc_dataset.image_path, 
                                            cfg.voc_dataset.train_txt_path, 
                                            cfg.training.is_augment)
            self.valid_dataset = VOCDataset(cfg.voc_dataset.anno_path, 
                                            cfg.voc_dataset.image_path, 
                                            cfg.voc_dataset.val_txt_path, 
                                            cfg.valid.is_augment)
        if stage == "test":
            pass
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return  DataLoader(self.train_dataset,
                            batch_size=cfg.training.batch_size, 
                            shuffle=cfg.training.shuffle,
                            num_workers=cfg.training.num_workers,
                            pin_memory=cfg.training.pin_memory)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_dataset,
                            batch_size=cfg.valid.batch_size, 
                            shuffle=cfg.training.shuffle,
                            num_workers=cfg.valid.num_workers,
                            pin_memory=cfg.valid.pin_memory)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass