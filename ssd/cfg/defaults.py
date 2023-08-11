from easydict import EasyDict


class Configuration:
    
    models = EasyDict({
        'image_size': [300, 300],
    })

    default_boxes = EasyDict({
        'respect_ratio': [1, 2, 3, 1/2, 1/3],
        's_min': 0.2,
        's_max': 0.9,
        'fm_sizes': [19, 19, 10, 5, 3, 1]
    })