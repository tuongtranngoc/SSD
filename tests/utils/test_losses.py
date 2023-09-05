from .. import *

import torch
from torch.utils.data import DataLoader

def test():
    loss = DataUtils.to_device(SSDLoss())
    train_dataset = COCODataset(label_path=cfg.dataset.train_label_path, image_path=cfg.dataset.train_img_path, is_augment=False)
    val_dataset = COCODataset(label_path=cfg.dataset.val_label_path, image_path=cfg.dataset.val_img_path, is_augment=False)
    dl = DataLoader(train_dataset, batch_size=20, shuffle=True)
    
    for ims, labels in dl:
        out1, out2 = loss(labels, labels)
        (out1 + out2).backward()
        print(out2.item(), out2.item())

if __name__ == "__main__":
    test()