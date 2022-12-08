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

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        # return the loss
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y, reduction="sum")

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y, reduction="sum")

        self.log("val_loss", loss)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y, reduction="sum")

        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])



class Transformer(Base):
    def __init__(
        self,
        head=3,
        kernel_number=512,
        kernel_length=7,
        filter_number=256,
        kernel_size=3,
        pooling_size=2,
        conv_layers=4,
        conv_repeat=1,
        hidden_size=256,
        dropout=0.2,
        h_layers=2,
        input_length=600 * 2,
        pooling_type="avg",
        padding="same",
        attention_layers=2,
        learning_rate=1e-3,
        num_rel_pos_features=66,
    ):
        super().__init__()
        self.save_hyperparameters()

        # This conv layer is appied on both forward and RC strands
        self.conv0 = ConvBlock(4, kernel_number, kernel_length, padding=padding)

        self.convlayers = nn.ModuleList()
        self.convlayers.append(
            ConvBlock(
                kernel_number,
                filter_number,
                kernel_size,
                padding=padding,
            )
        )

        fc_dim = input_length
        for layer in range(conv_layers):
            for repeat in range(conv_repeat):
                self.convlayers.append(
                    Residual(
                        ConvBlock(
                            filter_number,
                            filter_number,
                            kernel_size,
                            padding=padding,
                        )
                    )
                )
            if pooling_type == "max":
                self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            else:
                raise ValueError(
                    "Unknown pooling type, please choose from max, avg, attention, softmax"
                )

            # Calculate the fc_dim
            fc_dim = ceil(fc_dim / pooling_size)

        

        self.attentionlayers = nn.ModuleList()

        for layer in range(attention_layers):
            self.attentionlayers.append(
                nn.Sequential(
                    Residual(
                        Attention(
                            dim=filter_number,  # dimension of the last out channel
                            num_rel_pos_features=num_rel_pos_features,
                        ),
                    ),
                    nn.LayerNorm(filter_number),
                    Residual(
                        nn.Sequential(
                            nn.Linear(filter_number, filter_number * 2),
                            nn.Dropout(dropout),
                            nn.ReLU(),
                            nn.Linear(filter_number * 2, filter_number),
                            nn.Dropout(dropout),
                        )
                    ),
                    nn.LayerNorm(filter_number),
                )
            )


        self.fc0 = nn.Sequential(nn.Linear(fc_dim * filter_number, hidden_size), GELU())

        self.fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), GELU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers)
        )

        self.out = nn.Linear(hidden_size, 164)


    def forward(self, x):
        x = self.conv0(x)

        for layer in self.convlayers:
            x = layer(x)

        x = torch.permute(x, (0, 2, 1)) # (batch_size, length, filter_number)

        # Attention layer
        for layer in self.attentionlayers:
            x = layer(x)

        # flatten
        x = x.flatten(1)
        x = self.fc0(x)
        for layer in self.fclayers:
            x = layer(x)

        x = torch.sigmoid(self.out(x))
        return x