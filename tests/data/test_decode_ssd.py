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
    defbug_idxs = [i for i in range(0,6)]
    debug_dfbox_dir = cfg.debug.dfboxes

    for i in range(5):
        im_pth, bboxes = ds[i]
        im = cv2.imread(im_pth)
        for box in bboxes:
            im = cv2.rectangle(im, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), color=(0, 0, 255), thickness=2)
        im = cv2.resize(im, (cfg.models.image_size, cfg.models.image_size))

        dfboxes_fm1 = DefaultBoxesGenerator.build_default_boxes()[fm_sizes[4]].reshape(-1, 4)[defbug_idxs[0]:defbug_idxs[-1]]
        pred_bboxes = dfboxes_fm1.clone()

        pred_bboxes = BoxUtils.decode_ssd(pred_bboxes, dfboxes_fm1)
        pred_bboxes = BoxUtils.xcycwh_to_xyxy(pred_bboxes)
        pred_bboxes = BoxUtils.denormalize_box(pred_bboxes)
        pred_bboxes = pred_bboxes.detach().cpu().numpy()

        for box in pred_bboxes:
            im = cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(255, 0, 0), thickness=1)
        
        import pdb; pdb.set_trace()




if __name__ == "__main__":
    test()