from . import *

import torch
from torchsummary import summary
from torchview import draw_graph

def test():
    device = 'cuda'
    x = torch.randn(size=(3, 300, 300)).to(device)
    ssd = SSDModel().to(device)
    summary(ssd, x.shape, 1)
    draw_graph(ssd, input_size=x.shape, expand_nested=True, save_graph=True, directory='outputs', graph_name='ssd_vgg16')


if __name__ == "__main__":
    test()