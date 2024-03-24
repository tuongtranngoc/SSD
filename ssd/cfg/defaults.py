from easydict import EasyDict
import torch
import os

class Configuration:
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = EasyDict({
        'arch_name': 'vgg16_bn',
        'pretrained': True,
        'image_size': 300,
        'fm_channels': [512, 1024, 512, 256, 256, 256]
    })
    
    default_boxes = EasyDict({
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'strides': [8, 16, 32, 64, 100, 300],
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        's_min': 0.2,
        's_max': 0.9,
        'fm_sizes': [38, 19, 10, 5, 3, 1],
        'num_dfboxes': [4, 6, 6, 6, 4, 4],
        'iou_thresh': 0.35,
        'ratio_pos_neg': 3,
        'alpha': 1,
        'label_smooth': 0.1,
        'standard_norms': [0.2, 0.1]
    })

    voc_dataset = EasyDict({
        'num_classes': 21,
        'mean': [0.485, 0.456, 0.406], 
        'std': [0.229, 0.224, 0.225],
        'classes': 'dataset/VOC/classes.txt',
        'image_path': 'dataset/VOC/images',
        'anno_path': 'dataset/VOC/labels',
        'train_txt_path':["dataset/VOC/images_id/trainval2007.txt", "dataset/VOC/images_id/trainval2012.txt"],
        'val_txt_path': ['dataset/VOC/images_id/test2007.txt']
    })

    training = EasyDict({
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 8,
        'pin_memory': False,
        'is_augment': True,
        'lr': 1e-4,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'epochs': 121,
    })


    valid = EasyDict({
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
        'is_augment': False,
        'eval_step': 5,
    })
    
    debug = EasyDict({
        "tensorboard_debug": "exps/tensorboard",
        "training_debug": "exps/training",
        "dataset_debug": "exps/dataset",
        "valid_debug": "exps/valid",
        "test_cases": "exps/test_cases",
        "prediction": "exps/prediction",
        "ckpt_dirpath": "ssd/weights",
        "arch_model": "outputs",
        "idxs_debug": [20, 21, 22, 23, 24, 25, 26, 27],
        "augmentation_debug": "exps/augmentation",
        "log_file": "logs/ssd.log",
        "dfboxes": 'exps/dfboxes',
        "iou_thresh": 0.3,
        "conf_thresh": 0.6,
        "matched_dfboxes": "exps/matched_dfboxes",
        "dfboxes_generator": "exps/dfboxes_generator",
        "debug_mode": True,
    })