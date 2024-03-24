from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything

from . import *

from .pl_model import LitSSD
from .pl_dataset import VocModule


def main():
    model = LitSSD()
    data = VocModule()
    seed_everything(96, workers=True)
    torch.set_float32_matmul_precision('high')
    trainer = Trainer(
        max_epochs=30,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    trainer.fit(model=model, datamodule=data)

if __name__ == "__main__":
    
    main()
        