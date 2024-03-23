# main.py
# ! pip install torchvision
import pytorch_lightning as L
import torch
import torch.utils.data as data
import torchvision as tv
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT

from models.autoencoder_cnn import CnnAutoEncoder
from models.autoencoder_fcnn import FcnnAutoEncoder

print(
    "Cuda support:",
    torch.cuda.is_available(),
    ":",
    torch.cuda.device_count(),
    "devices",
)

# Step 2: Define data
# -------------------
dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
train, val = data.random_split(dataset, [0.8, 0.2])

# Step 3: Model
# -------------------
autoencoder = FcnnAutoEncoder(training_dataset_size=len(train))
autoencoder.to("cuda")

# Step 4: Train
# -------------------
# TensorBoardLogger : Create and add to your LightningModule
tb_logger = L.loggers.TensorBoardLogger("tb_logs", name="AE-MNIST")
trainer = L.Trainer(max_epochs=10, logger=tb_logger)
trainer.fit(autoencoder, data.DataLoader(train, num_workers=3), data.DataLoader(val))

# Step 5: Test
# -------------------
for i in range(5):
    trainer.test(autoencoder, data.DataLoader(val))
