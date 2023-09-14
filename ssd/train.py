from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import glob
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import *
from .eval import SSDEvaluate

logger = Logger.get_logger("TRAINING")


class Trainer:
    def __init__(self, args) -> None:
        self.args = args
        self.start_epoch = 1
        self.best_map50 = 0.0
        self.create_model()
        self.create_data_loader()
        self.eval = SSDEvaluate(self.valid_dataset, self.model)

    def create_data_loader(self):
        self.train_dataset = VOCDataset(cfg.voc_dataset.anno_path, cfg.voc_dataset.image_path, cfg.voc_dataset.train_txt_path, cfg.training.is_augment)
        self.valid_dataset = VOCDataset(cfg.voc_dataset.anno_path, cfg.voc_dataset.image_path, cfg.voc_dataset.val_txt_path, cfg.valid.is_augment)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=cfg.training.batch_size, 
                                       shuffle=cfg.training.shuffle,
                                       num_workers=cfg.training.num_workers,
                                       pin_memory=cfg.training.pin_memory)
        Visualizer.debug_dfboxes_generator(self.train_dataset, cfg.debug.idxs_debug)
        Visualizer.debug_matched_dfboxes(self.train_dataset, cfg.debug.idxs_debug)
        
    def create_model(self):
        self.model = SSDModel(arch_name=cfg.models.arch_name, pretrained=cfg.models.pretrained).to(cfg.device)
        self.loss_fn = SSDLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.training.lr, amsgrad=True)

        if self.args.resume:
            logger.info("Resuming training ...")
            last_ckpt = os.path.join(cfg.debug.ckpt_dirpath, self.args.model_type, 'last.pt')
            if os.path.exists(last_ckpt):
                ckpt = torch.load(last_ckpt, map_location=cfg.device)
                self.start_epoch = self.resume_training(ckpt)
                logger.info(f"Loading checkpoint with start epoch: {self.start_epoch}, best mAP_50: {self.best_map50}")

    
    def train(self):
        for epoch in range(self.start_epoch, cfg.training.epochs):
            mt_reg_loss = BatchMeter()
            mt_cls_loss = BatchMeter()

            for bz, (images, labels, _) in enumerate(self.train_loader):
                self.model.train()
                images = DataUtils.to_device(images)
                labels = DataUtils.to_device(labels)
                out = self.model(images)

                reg_loss, cls_loss = self.loss_fn(labels, out)
                total_loss = reg_loss + cls_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                mt_reg_loss.update(reg_loss.item())
                mt_cls_loss.update(cls_loss.item())
                
                print(f"Epoch {epoch} Batch {bz+1}/{len(self.train_loader)}, reg_loss: {mt_reg_loss.get_value(): .5f}, class_loss: {mt_cls_loss.get_value():.5f}", end="\r")

                Tensorboard.add_scalars("train_loss",
                                        epoch,
                                        reg_loss=mt_reg_loss.get_value('mean'),
                                        cls_loss=mt_cls_loss.get_value('mean'))
                        
            logger.info(f"Epoch: {epoch} - reg_loss: {mt_reg_loss.get_value('mean'): .5f}, cls_loss: {mt_cls_loss.get_value('mean'): .5f}")

            if epoch % cfg.valid.eval_step == 0:
                metrics = self.eval.evaluate()
                Tensorboard.add_scalars("eval_loss",
                                        epoch,
                                        reg_loss=metrics["eval_reg_loss"].get_value("mean"),
                                        cls_loss=metrics["eval_cls_loss"].get_value("mean"))
                
                Tensorboard.add_scalars("eval_mAP",
                                        epoch,
                                        map=metrics["eval_map"].get_value("mean"),
                                        map_50=metrics["eval_map_50"].get_value("mean"),
                                        map_75=metrics["eval_map_75"].get_value("mean"))
                
                # Save best checkpoint
                current_map50 = metrics["eval_map_50"].get_value("mean")
                if current_map50 > self.best_map50:
                    self.best_map50 = current_map50
                    best_cpkt_pth = os.path.join(cfg.debug.ckpt_dirpath, self.args.model_type, 'best.pt')
                    self.save_ckpt(best_cpkt_pth, self.best_map50, epoch)

            # Save last checkpoint
            last_ckpt = os.path.join(cfg.debug.ckpt_dirpath, self.args.model_type, 'last.pt')
            self.save_ckpt(last_ckpt, self.best_map50, epoch)

            # Debug after each training epoch
            Visualizer.debug_output(self.train_dataset, cfg.debug.idxs_debug, self.model, 'train', cfg.debug.training_debug, apply_nms=True)
            Visualizer.debug_output(self.valid_dataset, cfg.debug.idxs_debug, self.model, 'valid', cfg.debug.valid_debug, apply_nms=True)

    def save_ckpt(self, save_path, best_acc, epoch):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ckpt_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_map50": best_acc,
            "epoch": epoch,
        }
        logger.info(f"Saving checkpoint to {save_path}")
        torch.save(ckpt_dict, save_path)

    def resume_training(self, ckpt):
        self.best_map50 = ckpt['best_map50']
        start_epoch = ckpt['epoch'] + 1
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.model.load_state_dict(ckpt['model'])

        return start_epoch


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='vgg16',
                        help='Model selection contain: vgg16, vgg16-bn, resnet18, resnet34, resnet50')
    parser.add_argument('--resume', nargs='?', const=True, default=False, 
                        help='Resume most recent training')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    trainer = Trainer(args)
    trainer.train()