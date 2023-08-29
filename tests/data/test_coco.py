from .. import cfg
from .. import COCODataset

from torch.utils.data import DataLoader

def test():
    train_dataset = COCODataset(label_path=cfg.dataset.train_label_path, image_path=cfg.dataset.train_img_path, is_augment=False)
    val_dataset = COCODataset(label_path=cfg.dataset.val_label_path, image_path=cfg.dataset.val_img_path, is_augment=False)
    dl = DataLoader(train_dataset, batch_size=20, shuffle=True)
    next(iter(dl))

if __name__ == "__main__":
    test()