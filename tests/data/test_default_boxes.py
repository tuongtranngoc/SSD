from .. import BoxUtils
from .. import BaseDataset
from .. import coco_cfg as cfg
from .. import default_cfg
from .. import DefaultBoxesGenerator

import cv2


def test():
    train_dataset = BaseDataset(label_path=cfg.train_label_path, image_path=cfg.train_img_path)
    ds = train_dataset.load_coco_dataset()
    fm_sizes = default_cfg.default_boxes.fm_sizes
    for i in range(5):
        im_pth, bboxes = ds[i]
        im = cv2.imread(im_pth)
        for box in bboxes:
            im = cv2.rectangle(im, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), color=(0, 0, 255), thickness=1)
        im = cv2.resize(im, (300, 300))
        dfboxes_fm1 = DefaultBoxesGenerator.build_default_boxes()[fm_sizes[0]].reshape(-1, 4)
        dfboxes_fm1 = BoxUtils.xcycwh_to_xyxy(dfboxes_fm1)
        dfboxes_fm1 = BoxUtils.denormalize_box(dfboxes_fm1)
        dfboxes_fm1 = dfboxes_fm1.detach().cpu().numpy()
        for box in dfboxes_fm1[3000: 3010]:
            im = cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(255, 0, 0), thickness=1)


if __name__ == "__main__":
    test()