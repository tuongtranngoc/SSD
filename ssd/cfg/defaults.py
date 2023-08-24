from easydict import EasyDict


class Configuration:
    
    models = EasyDict({
        'image_size': 300,
    })

    default_boxes = EasyDict({
        'respect_ratio': [1, 2, 3, 1/2, 1/3],
        's_min': 0.2,
        's_max': 0.9,
        'fm_sizes': [38, 19, 10, 5, 3, 1],
        'iou_thresh': 0.5,
        'pos_ratio': 3,
        'neg_ratio': 1,
        'alpha': 1,
        'label_smooth': 0.1
    })