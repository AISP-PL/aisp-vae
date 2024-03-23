"""
    Autoencoder model with convolutional layers for MNIST dataset

    A LightningModule (nn.Module subclass) defines a full *system*
    (ie: an LLM, diffusion model, autoencoder, or simple image classifier).
"""

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnAutoEncoder(L.LightningModule):
    """Autoencoder model for MNIST dataset"""

    def __init__(
        self,
        training_dataset_size: int,
    ):
        """Initialize the model"""
        super().__init__()
        self.training_dataset_size = training_dataset_size

        # Model : Encoder (input image is 28x28 pixels)
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=10, kernel_size=3, stride=3, padding=1
            ),
            nn.ReLU(),
            # 1x28x28 -> 10x10x10
            nn.MaxPool2d(kernel_size=3, stride=1),
            # 10x10x10 -> 10x8x8
            nn.Conv2d(
                in_channels=10, out_channels=10, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            # 10x8x8 -> 10x4x4
            nn.MaxPool2d(kernel_size=2, stride=1),
            # 10x4x4 -> 10x3x3
            # nn.Flatten(),
            # 10x3x3 -> 90
        )

        # Model : Decoder - conv transpose from 90 to 28x28 pixels.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=10, out_channels=10, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            # 10x3x3 -> 10x8x8
            nn.ConvTranspose2d(
                in_channels=10, out_channels=10, kernel_size=5, stride=2, padding=2
            ),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            # 10x8x8 -> 10x16x16
            nn.ConvTranspose2d(in_channels=10, out_channels=1, kernel_size=2, stride=3),
            nn.BatchNorm2d(1),
            # 10x16x16 -> 1x26x26
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=1),
            nn.Sigmoid(),
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

        # Forward pass : X -> Z -> Y
        z = self.encoder(x)
        y = self.decoder(z)

        # Loss : MSE Loss between each pixel
        # of the original image and the reconstructed image
        loss = F.mse_loss(y, x)
        self.log("Training/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        # Batch : Unpack and reshape
        x, _old_y = batch

        # Forward pass : X -> Z -> Y
        z = self.encoder(x)
        y = self.decoder(z)

        # Loss : MSE Loss between each pixel
        # of the original image and the reconstructed image
        valid_loss = F.mse_loss(y, x)
        self.log("Valid/loss", valid_loss)

        # Images together : X and Y
        image = torch.cat((x, y), dim=1) * 255
        image_int = image.type(torch.uint8).reshape([1, 2 * 28, 28])
        image_raw = image_int.cpu().numpy()

        self.logger.experiment.add_image(
            "Valid/images",
            image_raw,
            self.current_epoch * self.training_dataset_size + batch_idx,
        )

    def configure_optimizers(self):
        """Configure optimizer for training"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
