from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import random
import numpy as np
from collections import defaultdict

import torch
from . import *
import torch.nn.functional as F


class AnnotationTool:
    """Annotation: color, class ids
    """
    def __init__(self) -> None:
        self.class_names = defaultdict()
        with open(cfg.voc_dataset.classes, 'r') as f:
            for i, l in enumerate(f.readlines()):
                self.class_names[l.strip()] = i
        f.close()
        self.colors = {k: tuple([random.randint(0, 255) for _ in range(3)])
                    for k in self.class_names.keys()}

    def class2color(self, cls_name):
        self.colors['groundtruth'] = (0, 0, 255)
        self.colors['background'] = (128, 128, 128)
        self.colors['dfboxes'] = (255, 0, 0)
        return self.colors[cls_name]

    def id2class(self, cls_id):
        ids = {v:k for k, v in self.class_names.items()}
        return ids[cls_id]


class Visualizer:
    """ Visualize debug images from training and valid
    """
    thickness = 1
    lineType = cv2.LINE_AA
    cvt_ano = AnnotationTool()
    h, w = cfg.models.image_size, cfg.models.image_size
    dfboxes = DefaultBoxesGenerator.df_bboxes.to(cfg.device)
    
    @classmethod
    def unnormalize_box(cls, bboxes:np.ndarray):
        bboxes = bboxes.copy()
        bboxes[..., [0, 2]] *= cls.w
        bboxes[..., [1, 3]] *= cls.h
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]].clip(min=0.0, max=cls.w)
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]].clip(min=0.0, max=cls.h)
        return bboxes

    @classmethod
    def draw_objects(cls, image, bboxes, confs, labels, conf_thresh, type_obj=None, unnormalize=False):
        for bbox, conf, label in zip(bboxes, confs, labels):
            if conf >= conf_thresh:
                image = cls.single_draw_object(image, bbox, conf, label, type_obj, unnormalize)
        return image
    
    @classmethod
    def single_draw_object(cls, image, bbox, conf, label,  type_obj=None, unnormalize=False):
        if label == 0: return image
        if unnormalize:
            bbox = cls.unnormalize_box(bbox)
        label = cls.cvt_ano.id2class(label)
        if type_obj == 'GT':
            color = cls.cvt_ano.class2color('groundtruth')
            text = label
        elif type_obj == 'PRED':
            color = cls.cvt_ano.class2color(label)
            text = '-'.join([label, str(round(conf, 3))])
        elif type_obj == 'dfboxes':
            color = cls.cvt_ano.class2color('dfboxes')
            text = ''
        else:
            Exception(f"Not have type_obj is None")

        cv2.rectangle(image,
                    (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                    color=color,
                    thickness=cls.thickness,
                    lineType=cls.lineType)
        
        cv2.putText(image, text,
                    (int(bbox[0]), int(bbox[1] + 0.025*cls.w)),
                    fontFace=0,
                    fontScale=cls.thickness/3,
                    color=color,
                    thickness=cls.thickness,
                    lineType=cls.lineType)
        
        return image

    @classmethod
    def debug_output(cls, dataset, idxs, model, type_fit, debug_dir, apply_nms=True):
        os.makedirs(os.path.join(debug_dir, type_fit), exist_ok=True)
        model.eval()
        for i, idx in enumerate(idxs):
            img_path, targets = dataset.voc_dataset[idx]
            target_labels, target_bboxes = targets[..., 0], targets[..., 1:]
            target_confs = np.ones_like(target_labels, dtype=np.float32)
            # Normalize bboxes
            image, target_bboxes, target_labels = dataset.get_image_label(img_path, target_bboxes, target_labels, False)
            target_bboxes =  torch.tensor(target_bboxes, dtype=torch.float32, device=cfg.device)
            target_bboxes = BoxUtils.normalize_box(target_bboxes)
            target_confs =  torch.tensor(target_confs, dtype=torch.float32, device=cfg.device)
            target_labels = torch.tensor(target_labels, dtype=torch.long, device=cfg.device)
            # Decode bboxes
            pred_bboxes, pred_confs = model(image.to(cfg.device).unsqueeze(0))
            pred_bboxes, pred_confs = pred_bboxes.squeeze(0), pred_confs.squeeze(0)
            pred_bboxes = BoxUtils.decode_ssd(pred_bboxes, cls.dfboxes)
            pred_bboxes = BoxUtils.xcycwh_to_xyxy(pred_bboxes)
            
            pred_confs = torch.softmax(pred_confs, dim=-1)
            confs, cates = pred_confs.max(dim=-1)
            # Filter negative predictions
            pred_pos_mask = cates > 0
            pred_bboxes = pred_bboxes[pred_pos_mask]
            confs = confs[pred_pos_mask]
            cates = cates[pred_pos_mask]
            # Apply non-max suppression
            if apply_nms:
                pred_bboxes, confs, cates = BoxUtils.nms(pred_bboxes, confs, cates, cfg.debug.iou_thresh, cfg.debug.conf_thresh)
            # Tensor to numpy
            target_bboxes, target_confs, target_labels = DataUtils.to_numpy([target_bboxes, target_confs, target_labels])
            pred_bboxes, confs, cates = DataUtils.to_numpy([pred_bboxes, confs, cates])
            image = DataUtils.image_to_numpy(image)
            # Visualize debug images
            image = cls.draw_objects(image, target_bboxes, target_confs, target_labels, cfg.debug.conf_thresh, type_obj='GT', unnormalize=True)
            image = cls.draw_objects(image, pred_bboxes, confs, cates, cfg.debug.conf_thresh, type_obj='PRED', unnormalize=True)
            
            cv2.imwrite(os.path.join(debug_dir, type_fit, f'{i}.png'), image)
    
    @classmethod
    def debug_matched_dfboxes(cls, dataset, idxs):
        os.makedirs(cfg.debug.matched_dfboxes, exist_ok=True) 
        for i, idx in enumerate(list(range(100))):
            img_path, targets = dataset.voc_dataset[idx]
            target_labels, target_bboxes = targets[..., 0], targets[..., 1:]
            target_confs = np.ones_like(target_labels, dtype=np.float32)
            __ , matched_dfboxes, __ = dataset[idx]
            df_bboxes, df_labels = matched_dfboxes
            # Normalize bboxes
            image, target_bboxes, target_labels = dataset.get_image_label(img_path, target_bboxes, target_labels, False)
            # Filter nagative predictions
            pos_mask = df_labels > 0
            df_labels = DataUtils.single_to_numpy(df_labels[pos_mask])
            df_bboxes = BoxUtils.decode_ssd(df_bboxes[pos_mask], cls.dfboxes[pos_mask])
            df_bboxes = BoxUtils.xcycwh_to_xyxy(df_bboxes)
            df_bboxes = DataUtils.single_to_numpy(df_bboxes)
            df_confs = np.ones_like(df_labels, np.float32)
            # Visualize debug
            image = DataUtils.image_to_numpy(image)
            image = cls.draw_objects(image, df_bboxes, df_confs, df_labels, cfg.debug.conf_thresh, type_obj='dfboxes', unnormalize=True)
            image = cls.draw_objects(image, target_bboxes, target_confs, target_labels, cfg.debug.conf_thresh, type_obj='GT', unnormalize=False)
            cv2.imwrite(os.path.join(cfg.debug.matched_dfboxes, f'{idx}.png'), image)
    
    @classmethod
    def debug_dfboxes_generator(cls, dataset, idxs):
        os.makedirs(cfg.debug.dfboxes_generator, exist_ok=True)
        fm_sizes = cfg.default_boxes.fm_sizes
        dfboxes_fm = DefaultBoxesGenerator.build_default_boxes()
        
        for i in idxs:
            im_pth, _ = dataset.voc_dataset[i]
            im = cv2.imread(im_pth)
            im = cv2.resize(im, (cfg.models.image_size, cfg.models.image_size))
            id_pth = os.path.join(cfg.debug.dfboxes_generator)
            os.makedirs(id_pth, exist_ok=True)
            for fm in fm_sizes:
                im_fm = im.copy()
                pos_dfboxes = fm//2
                # import ipdb; ipdb.set_trace();
                dfboxes = dfboxes_fm[fm][pos_dfboxes, pos_dfboxes, ...].reshape(-1, 4)
                dfboxes = BoxUtils.xcycwh_to_xyxy(dfboxes)
                dfboxes = BoxUtils.denormalize_box(dfboxes)
                dfboxes = dfboxes.detach().cpu().numpy()
                for box in dfboxes:
                    im_fm = cv2.rectangle(im_fm, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 0, 255), thickness=cls.thickness)
                cv2.imwrite(os.path.join(id_pth, f'fm_{fm}.png'), im_fm)
    
    @classmethod
    def debug_augmentation(cls, dataset):
        os.makedirs(cfg.debug.augmentation_debug, exist_ok=True)
        range_idxs = list(range(0, 30))
        aug = AlbumAug()
        anno_cvt = AnnotationTool()
        for idx in range_idxs:
            im_pth, targets = dataset.voc_dataset[idx]
            im = cv2.imread(im_pth)
            labels, bboxes = targets[..., 0], targets[..., 1:]
            im, bboxes, labels = aug(im, bboxes, labels)
            for box, label in zip(bboxes, labels):
                im = cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(255, 0, 0), thickness=cls.thickness)
                im = cv2.putText(im, str(anno_cvt.id2class(int(label))), (int(box[0]), int(box[1]+0.025*cls.w)), fontFace=0, fontScale=cls.thickness/2, color=(255, 0, 0), thickness=cls.thickness)
            cv2.imwrite(os.path.join(cfg.debug.augmentation_debug, f'{os.path.basename(im_pth)}'), im)

    @classmethod
    def debug_arch_model(cls, model):
        from torchview import draw_graph
        x = torch.randn(size=(3, cfg.models.image_size, cfg.models.image_size)).to(cfg.device)
        ssd = model.to(cfg.device)
        draw_graph(ssd, input_size=x.unsqueeze(0).shape, 
                   expand_nested=True, 
                   save_graph=True, 
                   directory=cfg.debug.arch_model, 
                   graph_name=cfg.models.arch_name)
        
    @classmethod
    def debug_negative(cls, idxs, preds):
        pass