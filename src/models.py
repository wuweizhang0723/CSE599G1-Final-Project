from math import ceil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from src.layers import (
    ConvBlock,
    GELU,
    Residual,
    Attention,
)


class Base(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x):
        return NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                # TODO: add an argument to control the patience
                optimizer,
                patience=2,
            ),
            "monitor": "val_loss",
        }

        return [optimizer], [lr_scheduler]

    def trainning_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        # return the loss
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y, reduction="sum")

        return loss

    def validation_step(self, batch, batch_idx):

        return NotImplementedError

    def test_step(self, batch, batch_idx):
        return NotImplementedError