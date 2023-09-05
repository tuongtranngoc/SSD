from .. import *

import os
import cv2

def test():
    os.makedirs(cfg.debug.augmentation_debug, exist_ok=True)
    coco_anno = COCOAnnotation()
    aug = AlbumAug()
    train_dataset = BaseDataset(label_path=cfg.dataset.train_label_path, image_path=cfg.dataset.train_img_path)
    ds = train_dataset.load_coco_dataset()
    for img_info in ds[:10]:
        img_pth, targets = img_info
        img = cv2.imread(img_pth)
        labels, bboxes = targets[:, 0], targets[:, 1:]
        img, bboxes, labels = aug(img, bboxes, labels)

        for box, label in zip(bboxes, labels):
            img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(255, 0, 0), thickness=1)
            img = cv2.putText(img, str(coco_anno.id2class(int(label))), (int(box[0]), int(box[1])+5), fontFace=0, fontScale=1/3, color=(255, 0, 0), thickness=1)
        
        cv2.imwrite(os.path.join(cfg.debug.augmentation_debug, f'{os.path.basename(img_pth)}'), img)


if __name__ == "__main__":
    test()