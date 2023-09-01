from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import torch
from collections import defaultdict

from ..utils import cfg


class DefaultBoxesGenerator:
    m = cfg.default_boxes.fm_sizes
    s_min = cfg.default_boxes.s_min
    s_max = cfg.default_boxes.s_max
    ratios = cfg.default_boxes.respect_ratio
    im_size = cfg.models.image_size
    
    @classmethod
    def build_default_boxes(cls):
        df_bboxes = defaultdict()
        
        for i, fm_size in enumerate(cls.m):
            k = i + 1
            df_bboxes[fm_size] = torch.zeros(size=(fm_size, fm_size, 6 , 4))
            
            idxs_i = torch.arange(fm_size)
            idxs_j = torch.arange(fm_size)
            pos_j, pos_i = torch.meshgrid(idxs_j, idxs_i, indexing='ij')
            
            xc = (pos_i + 0.5) / fm_size
            yc = (pos_j + 0.5) / fm_size
            
            xcyc = torch.stack((xc, yc), dim=-1)
            xcyc = xcyc.unsqueeze(2).expand((-1, -1, 6, -1))
            df_bboxes[fm_size][..., :2] = xcyc
            
            wh_ratios = []
            s_k = cls.s_min + (cls.s_max - cls.s_min) * (k - 1) / (len(cls.m) - 1)
            for a_r in cls.ratios:
                if a_r == 1:
                    s_k_1 = cls.s_min + (cls.s_max - cls.s_min) * (k + 1 - 1) / (len(cls.m) - 1)
                    s_prime_k = math.sqrt(s_k * s_k_1)
                    wh_ratios.extend([[s_k, s_k], [s_prime_k, s_prime_k]])
                else:
                    w_k = s_k * math.sqrt(a_r)
                    h_k = s_k / math.sqrt(a_r)
                    wh_ratios.append([w_k, h_k])

            wh_ratios = torch.tensor(wh_ratios, dtype=torch.float32)
            wh_ratios = torch.clamp(wh_ratios, min=0.0, max=1.0)
            df_bboxes[fm_size][..., 2:] = wh_ratios

        return df_bboxes
    
    @classmethod
    def merge_defaultboxes(cls, dfboxes_dict:dict):
        default_boxes = []
        for fm, dfbox in dfboxes_dict.items():
            dfbox = dfbox.reshape(-1, 4)
            default_boxes.append(dfbox)
        default_boxes = torch.cat(default_boxes, dim=0)
        return default_boxes


if __name__ == "__main__":
    DefaultBoxesGenerator.build_default_boxes()    