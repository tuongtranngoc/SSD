from . import *
import torch

def test():
    x = torch.randn((1, 3, 300, 300))
    ssd = SSD()
    out = ssd(x)

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    test()