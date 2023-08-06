import torch
import torchvision
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}


class VGG16(nn.Module):
    def __init__(self, batch_norm=False) -> None:
        super().__init__()
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M_ceil_mode', 512, 512, 512]
        self.features = self.make_layers(cfg, batch_norm)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'M_ceil_mode':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]

                in_channels = v
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        return x
    

def build_backbone(arch_name='vgg16', pretrained=True):
    feat_dims = 512
    if arch_name == "vgg16":
        model = VGG16()
    elif arch_name == "vgg16-bn":
        model = VGG16(batch_norm=True)
    else:
        RuntimeError("Only support model in [vgg16, vgg16-bn]")
    
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[arch_name]), strict=False)
    
    return model, feat_dims