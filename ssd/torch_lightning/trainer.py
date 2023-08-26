from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytorch_lightning as pl


class LitSSDModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

    
