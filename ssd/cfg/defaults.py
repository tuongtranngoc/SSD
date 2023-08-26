from easydict import EasyDict
import torch

class Configuration:
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = EasyDict({
        'arch_name': 'vgg16',
        'pretrained': True,
        'image_size': 300,
        'fm_channels': [512, 1024, 512, 256, 256, 256]
    })
    
    default_boxes = EasyDict({
        'respect_ratio': [1, 2, 3, 1/2, 1/3],
        's_min': 0.2,
        's_max': 0.9,
        'fm_sizes': [38, 19, 10, 5, 3, 1],
        'dfboxes_sizes': [6, 6, 6, 6, 6, 6],
        'iou_thresh': 0.5,
        'pos_ratio': 3,
        'neg_ratio': 1,
        'alpha': 1,
        'label_smooth': 0.1,
    })

    dataset = EasyDict({
    'num_classes': 80,
    'train_img_path': 'dataset/COCO/train2017',
    'train_label_path': 'dataset/COCO/annotations/instances_train2017.json',
    'val_img_path': 'dataset/COCO/val2017',
    'val_label_path': 'dataset/COCO/annotations/instances_val2017.json'
    })

    training = EasyDict({
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 8,
        'pin_memory': True,
        'is_augment': True,
        'lr': 1e-4,
        'epochs': 150
    })


    valid = EasyDict({
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
        'is_augment': False
    })