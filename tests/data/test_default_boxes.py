from .. import BoxUtils
from .. import BaseDataset
from .. import cfg
from .. import DefaultBoxesGenerator

import cv2
import os


def test():
    train_dataset = BaseDataset(label_path=cfg.dataset.train_label_path, image_path=cfg.dataset.train_img_path)
    ds = train_dataset.load_coco_dataset()
    fm_sizes = cfg.default_boxes.fm_sizes
    defbug_idxs = [i for i in range(3000,3010)]
    debug_dfbox_dir = cfg.debug.dfboxes
    
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
        for box in dfboxes_fm1[defbug_idxs]:
            im = cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(255, 0, 0), thickness=1)
        cv2.imwrite(os.path.join(debug_dfbox_dir, f'{i}.png'), im)


if __name__ == "__main__":
    test()