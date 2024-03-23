"""
    Autoencoder model with fully connected layers

    A LightningModule (nn.Module subclass) defines a full *system*
    (ie: an LLM, diffusion model, autoencoder, or simple image classifier).
"""

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class FcnnAutoEncoder(L.LightningModule):
    """Autoencoder model for MNIST dataset"""

    def __init__(
        self,
        training_dataset_size: int,
    ):
        """Initialize the model"""
        super().__init__()
        self.training_dataset_size = training_dataset_size
        # Model : Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 32)
        )
        # Model : Decoder (upscaling the image back to 56x56 pixels)
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.ReLU(),
            nn.Linear(28 * 28, 56 * 56),
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
        x = x.view(x.size(0), -1)

        # Forward pass : X -> Z -> Y
        z = self.encoder(x)
        y_upscaled = self.decoder(z)

        # For loss, we have to downscale y to 28x28 from 56x56
        y = y_upscaled.reshape([1, 1, 56, 56])
        y = F.interpolate(y, size=(28, 28), mode="bilinear")
        y = y.reshape([1, -1])

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

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        # Batch : Unpack and reshape
        x, _old_y = batch
        x = x.view(x.size(0), -1)

        # Forward pass : X -> Z -> Y
        z = self.encoder(x)
        y_upscaled = self.decoder(z)

        # For loss, we have to downscale y to 28x28 from 56x56
        y = y_upscaled.reshape([1, 1, 56, 56])
        y = F.interpolate(y, size=(28, 28), mode="bilinear")
        y = y.reshape([1, -1])

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

        # Image upscaled : y_upscaled
        image_upscaled = y_upscaled * 255
        image_upscaled_int = image_upscaled.type(torch.uint8).reshape([1, 56, 56])
        image_upscaled_raw = image_upscaled_int.cpu().numpy()

        self.logger.experiment.add_image(
            "Valid/images_upscaled",
            image_upscaled_raw,
            self.current_epoch * self.training_dataset_size + batch_idx,
        )

    def configure_optimizers(self):
        """Configure optimizer for training"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
