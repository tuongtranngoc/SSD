from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import torch
import numpy as np
from collections import defaultdict


from ..utils import cfg

class DefaultBoxes:

    @classmethod
    def build_default_boxes(self):
        m = cfg.default_boxes.fm_sizes
        s_min = cfg.default_boxes.s_min
        s_max = cfg.default_boxes.s_max
        ratios = cfg.default_boxes.respect_ratio

        df_bboxes = defaultdict()

        for k in m:
            df_bboxes[k] = torch.zeros(size=(k, k, 6 , 4))

            idxs_i = torch.arange(k)
            idxs_j = torch.arange(k)
            pos_j, pos_i = torch.meshgrid(idxs_j, idxs_i, indexing='ij')

            xc = (pos_i + 0.5) / k
            yc = (pos_j + 0.5) / k

            xcyc = torch.stack((xc, yc), dim=-1)
            xcyc = xcyc.unsqueeze(2).expand((-1, -1, 6, -1))
            df_bboxes[k][..., :2] = xcyc

            wh_ratios = []
            s_k = s_min + (s_max - s_min) * (k - 1) / (len(m) - 1)
            for i, a_r in enumerate(ratios):
                if a_r == 1:
                    s_k_1 = s_min + (s_max - s_min) * (k + 1 - 1) / (len(m) - 1)
                    s_k_0 = (s_k * s_k_1)
                    w_k = s_k_0 * math.sqrt(a_r)
                    h_k = s_k_0 / math.sqrt(a_r)
                    wh_ratios.append([w_k, h_k])
                w_k = s_k * math.sqrt(a_r)
                h_k = s_k / math.sqrt(a_r)
                wh_ratios.append([w_k, h_k])

            df_bboxes[k][..., 2:] = torch.tensor(wh_ratios)
        return df_bboxes
    

if __name__ == "__main__":
    DefaultBoxes.build_default_boxes()    