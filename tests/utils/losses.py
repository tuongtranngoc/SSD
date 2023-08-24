from .. import coco_cfg as cfg
from .. import COCODataset
from .. import SSDLoss

import torch
from torch.utils.data import DataLoader

def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss = SSDLoss()
    train_dataset = COCODataset(label_path=cfg.train_label_path, image_path=cfg.train_img_path, is_augment=False)
    val_dataset = COCODataset(label_path=cfg.val_label_path, image_path=cfg.val_img_path, is_augment=False)
    dl = DataLoader(train_dataset, batch_size=20, shuffle=True)
    
    for ims, labels in dl:
        out1, out2 = loss(labels, labels)
        (out1 + out2).backward()
        print(out2.item(), out2.item())

if __name__ == "__main__":
    test()