from ssd.models.modules.backbone import VGG16, build_backbone
import torch

def test():
    x = torch.randn((1, 3, 300, 300))
    backbone, _ = build_backbone('vgg16', pretrained=True)
    out = backbone(x)
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    test()