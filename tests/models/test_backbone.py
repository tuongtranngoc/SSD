from . import vgg16_extractor, vgg16_bn_extractor
import torch.nn as nn
import torch

def test():
    x = torch.randn((1, 3, 300, 300))
    backbone = vgg16_extractor(pretrained=True)
    out = backbone(x)
    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    test()