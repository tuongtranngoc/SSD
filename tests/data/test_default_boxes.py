from . import DefaultBoxesGenerator
from . import BaseDataset, coco_cfg
from . import default_cfg as default_cfg

import numpy as np
import torch
import cv2


def test():
    im_size = 300
    im_fm = np.ones(shape=(im_size, im_size, 3)) * 255
    df_bboxes = DefaultBoxesGenerator.build_default_boxes()
    for k, df_bb in df_bboxes.items():
        if k != 3: continue
        grid_size = im_size // k
        for i in range(k):
            im_fm = cv2.line(im_fm, (grid_size * i, 0), (grid_size * i, im_size), (0, 0, 255), 1)
        for i in range(k):
            im_fm = cv2.line(im_fm, (0, i * grid_size), (im_size, i * grid_size), (0, 0, 255), 1)
        
        
        import pdb
        pdb.set_trace()




if __name__ == "__main__":
    test()