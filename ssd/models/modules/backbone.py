import torch
import torchvision
import torch.nn as nn


model_weights = {
    'vgg16': 'VGG16_Weights.IMAGENET1K_V1',
    'vgg16_bn': 'VGG16_BN_Weights.IMAGENET1K_V1',
    'resnet34': 'ResNet34_Weights.IMAGENET1K_V1'
}

def vgg16_extractor(arch_name):
    if arch_name == 'vgg16':
        return torchvision.models.vgg16(weights=model_weights[arch_name]).features
    elif arch_name == 'vgg16_bn':
        return torchvision.models.vgg16_bn(weights=model_weights[arch_name]).features
    elif arch_name == 'resnet34':
        return torchvision.models.resnet34(weigths=model_weights[arch_name]).features
    else:
        raise Exception("The model name is not available")