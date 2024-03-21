from itertools import product
from collections import defaultdict

import torch
from math import sqrt

from . import cfg

class DefaultBoxesGenerator:
    image_size = cfg.models['image_size']
    feature_maps = cfg.default_boxes['fm_sizes']
    min_sizes = cfg.default_boxes['min_sizes']
    max_sizes = cfg.default_boxes['max_sizes']
    strides = cfg.default_boxes['strides']
    num_dfboxes = cfg.default_boxes['num_dfboxes']
    aspect_ratios = cfg.default_boxes['aspect_ratios']
    clip = True

    @classmethod
    def build_default_boxes(cls):
        priors = defaultdict(list)
        for k, (f, s) in enumerate(zip(cls.feature_maps, cls.num_dfboxes)):
            scale = cls.image_size / cls.strides[k]
            for i, j in product(range(f), repeat=2):
                # unit center x,y
                cx = (j + 0.5) / scale
                cy = (i + 0.5) / scale

                # small sized square box
                size = cls.min_sizes[k]
                h = w = size / cls.image_size
                priors[f].append([cx, cy, w, h])

                # big sized square box
                size = sqrt(cls.min_sizes[k] * cls.max_sizes[k])
                h = w = size / cls.image_size
                priors[f].append([cx, cy, w, h])

                # change h/w ratio of the small sized box
                size = cls.min_sizes[k]
                h = w = size / cls.image_size
                for ratio in cls.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors[f].append([cx, cy, w * ratio, h / ratio])
                    priors[f].append([cx, cy, w / ratio, h * ratio])

        return priors
    
    @classmethod
    def merge_defaultboxes(cls, dfboxes_dict:dict):
        df_bboxes = []
        for __, dfbox in dfboxes_dict.items():
            dfbox = torch.tensor(dfbox)
            dfbox = dfbox.reshape(-1, 4)
            df_bboxes.append(dfbox)
        cls.df_bboxes = torch.cat(df_bboxes, dim=0)
        if cls.clip:
            cls.df_bboxes.clamp_(max=1, min=0)
        return cls.df_bboxes

dfboxes = DefaultBoxesGenerator.build_default_boxes()
DefaultBoxesGenerator.merge_defaultboxes(dfboxes)