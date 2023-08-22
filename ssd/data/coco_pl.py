from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytorch_lightning as pl

class COCODataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()