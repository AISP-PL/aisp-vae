# main.py
# ! pip install torchvision
from typing import Any
import numpy as np
from PIL import Image
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import TensorBoardLogger
import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
import pytorch_lightning as L

# --------------------------------
# Step 1: Define a LightningModule
# --------------------------------
# A LightningModule (nn.Module subclass) defines a full *system*
# (ie: an LLM, diffusion model, autoencoder, or simple image classifier).


class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("test_loss", loss)

        # Image : original
        x_array = x.reshape([28,28])
        original_raw = x_array.cpu().numpy() * 255
        original_raw = original_raw.astype(np.uint8).reshape([1,28,28])
        # original_image = Image.fromarray(original_raw)

        # Image : Create from x_hat
        out_array = x_hat.reshape([28,28])
        out_raw = out_array.cpu().numpy() * 255
        out_raw = out_raw.astype(np.uint8).reshape([1,28,28])
        # out_image = Image.fromarray(out_raw)

        self.logger.experiment.add_image("original", 
                                         original_raw, 
                                         self.current_epoch)
        self.logger.experiment.add_image("reconstruction", 
                                         out_raw, 
                                         self.current_epoch)



    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return super().validation_step(*args, **kwargs)

    def configure_optimizers(self):
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
tb_logger = L.loggers.TensorBoardLogger("tb_logs", 
                                        name="AE-MNIST")
trainer = L.Trainer(max_epochs=3, 
                        logger=tb_logger)
trainer.fit(autoencoder, 
            data.DataLoader(train), 
            data.DataLoader(val))

# -------------------
# Step 5: Test
# -------------------
for i in range(32):
    trainer.test(autoencoder, 
                data.DataLoader(val))