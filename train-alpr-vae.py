# main.py
# ! pip install torchvision
from typing import Any

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT

# --------------------------------
# Step 1: Define a LightningModule
# --------------------------------
# A LightningModule (nn.Module subclass) defines a full *system*
# (ie: an LLM, diffusion model, autoencoder, or simple image classifier).


class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 28 * 28)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        """
        Training step for single batch

        Parameters:
        -----------
        batch : Tuple of (x, y)
            x : Tensor
                Input tensor
            y : Tensor
                Target tensor
        batch_idx : int
            Index of the batch
        """
        # Batch : Unpack and reshape
        x, _old_y = batch
        x = x.view(x.size(0), -1)

        # Forward pass : X -> Z -> Y
        z = self.encoder(x)
        y = self.decoder(z)

        # Loss : MSE Loss between each pixel
        # of the original image and the reconstructed image
        loss = F.mse_loss(y, x)
        self.log("Training/loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Evaluation test step for single batch

        Parameters:
        -----------
        batch : Tuple of (x, y)
            x : Tensor
                Input tensor
            y : Tensor
                Target tensor
        batch_idx : int
            Index of the batch
        """
        # Batch : Unpack and reshape
        x, _old_y = batch
        x = x.view(x.size(0), -1)

        # Forward pass : X -> Z -> Y
        z = self.encoder(x)
        y = self.decoder(z)

        # Loss : MSE Loss between each pixel
        # of the original image and the reconstructed image
        loss = F.mse_loss(y, x)
        self.log("Test/loss", loss)

        # Image : original
        x_array = x.reshape([28, 28])
        original_raw = x_array.cpu().numpy() * 255
        original_raw = original_raw.astype(np.uint8).reshape([1, 28, 28])
        # original_image = Image.fromarray(original_raw)

        # Image : Create from y
        out_array = y.reshape([28, 28])
        out_raw = out_array.cpu().numpy() * 255
        out_raw = out_raw.astype(np.uint8).reshape([1, 28, 28])
        # out_image = Image.fromarray(out_raw)

        self.logger.experiment.add_image("Test/image", original_raw, self.current_epoch)
        self.logger.experiment.add_image("Test/image_out", out_raw, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        # Batch : Unpack and reshape
        x, _old_y = batch
        x = x.view(x.size(0), -1)

        # Forward pass : X -> Z -> Y
        z = self.encoder(x)
        y = self.decoder(z)

        # Loss : MSE Loss between each pixel
        # of the original image and the reconstructed image
        valid_loss = F.mse_loss(y, x)
        self.log("Valid/loss", valid_loss)

        # Image : original
        x_array = x.reshape([28, 28])
        original_raw = x_array.cpu().numpy() * 255
        original_raw = original_raw.astype(np.uint8).reshape([1, 28, 28])
        # original_image = Image.fromarray(original_raw)

        # Image : Create from y
        out_array = y.reshape([28, 28])
        out_raw = out_array.cpu().numpy() * 255
        out_raw = out_raw.astype(np.uint8).reshape([1, 28, 28])
        # out_image = Image.fromarray(out_raw)

        self.logger.experiment.add_image(
            "Valid/image", original_raw, self.current_epoch
        )
        self.logger.experiment.add_image("Valid/image_out", out_raw, self.current_epoch)

    def configure_optimizers(self):
        """Configure optimizer for training"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# -------------------
# Step 2: Define data
# -------------------
dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
train, val = data.random_split(dataset, [55000, 5000])

# -------------------
# Step 3: Model
# -------------------
autoencoder = LitAutoEncoder()

# -------------------
# Step 4: Train
# -------------------
# TensorBoardLogger : Create and add to your LightningModule
tb_logger = L.loggers.TensorBoardLogger("tb_logs", name="AE-MNIST")
trainer = L.Trainer(max_epochs=5, logger=tb_logger)
trainer.fit(autoencoder, data.DataLoader(train), data.DataLoader(val))

# -------------------
# Step 5: Test
# -------------------
for i in range(5):
    trainer.test(autoencoder, data.DataLoader(val))
