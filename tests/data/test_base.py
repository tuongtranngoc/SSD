from .. import cfg
from .. import BaseDataset

def test():
    train_dataset = BaseDataset(label_path=cfg.train_label_path, image_path=cfg.train_img_path)
    val_dataset = BaseDataset(label_path=cfg.val_label_path, image_path=cfg.val_img_path)
    print(f"Shape of train dataset: {len(train_dataset.load_coco_dataset())}")
    print(f"Shape of val dataset: {len(val_dataset.load_coco_dataset())}")


if __name__ == "__main__":
    test()