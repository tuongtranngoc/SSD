from easydict import EasyDict

coco = EasyDict({
    'train_img_path': 'dataset/COCO/train2017',
    'train_label_path': 'dataset/COCO/annotations/instances_train2017.json',
    'val_img_path': 'dataset/COCO/val2017',
    'val_label_path': 'dataset/COCO/annotations/instances_val2017.json'
})