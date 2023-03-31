import os
import json
import math
import numpy as np

import pandas

## Imports for plotting
import matplotlib.pyplot as plt
#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()
sns.set()

# ## Progress bar
# from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import time
import pandas as pd
import numpy as np
import itertools

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import random_split, DataLoader

# load subset of data used to train bilinear model
subset_df = pd.read_csv("subset_data.csv") 
subset_df = subset_df.iloc[: , 1:]

# one hot encoding the Pedigree 
Pedigree = subset_df.iloc[: , 1]
Pedigree = Pedigree.values
Pedigree = Pedigree.reshape(-1,1)
encoder = OneHotEncoder()
one_hot = encoder.fit_transform(Pedigree)
seed_one_hot = one_hot.toarray()
subset_df["Seed"] = list(seed_one_hot)

# extract field names, seed names and field x seed name combinations
field_id = subset_df['Field-Location_x'].unique().tolist()
seed_id = subset_df['Pedigree_x'].unique().tolist()
seedxfield = list(itertools.product(seed_id, field_id))
print('Example of seed and field combination:', seedxfield[0])

# build dataframe with number of samples regarding to the field and seed 
unique_seed_field = subset_df.groupby(['Field-Location_x', 'Pedigree_x']).size().reset_index().rename(columns={0:"count"})

# generating list of valid seed x field combinations
valid_seedxfield = unique_seed_field[["Field-Location_x","Pedigree_x"]].values

digits = load_digits()
data = digits.data
labels = digits.target

obs_raw = subset_df[['Loc-ID-Onehot','Seed','% Sand', '% Silt', '% Clay', 'v_daily_avg_temp', 'v_daily_avg_radiation', 'v_daily_avg_photo_p']]
obs_raw = obs_raw.to_numpy()

for i in range(len(obs_raw)):
    obs_raw[i][0] = obs_raw[i][0].strip('][').split(', ')

obs = []
for i in range(len(obs_raw)):
  obs.append(list([int(id) for id in obs_raw[i][0]])  + list(obs_raw[i][1])  + list(obs_raw[i][2:]))
obs = np.asarray(obs)

labels = subset_df['Plot Yield (ibs/ft2)'].values

obs = torch.tensor(obs)
obs = obs.type(torch.float)

# Define the sizes of the training, validation, and test sets
train_size = int(0.8 * len(obs))
val_size = int(0.1 * len(obs))
test_size = len(obs) - train_size - val_size

# Split the dataset into training, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(obs, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def get_train_data(num):
    return torch.stack([train_dataset[i] for i in range(num)])

# Set device to use GPU if available, else use CPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "/mnt/raid10/Bandits/data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "/mnt/raid10/Bandits/models"

class Encoder(nn.Module):
    def __init__(self, 
                num_input_size:int,  
                latent_dim:int, 
                act_fn: object = nn.ReLU):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            nn.Linear(in_features= num_input_size, out_features=128),
            act_fn(),
            nn.Linear(in_features=128, out_features=64),
            act_fn(),
            nn.Linear(in_features=64, out_features=36),
            act_fn(),
            nn.Linear(in_features=36, out_features=18),
            act_fn(),
            nn.Linear(in_features=18, out_features=latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self,
                num_input_size : int,
                latent_dim : int, 
                act_fn: object = nn.ReLU):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 18),
            act_fn(),
            torch.nn.Linear(18, 36),
            act_fn(),
            torch.nn.Linear(36, 64),
            act_fn(),
            torch.nn.Linear(64, 128),
            act_fn(),
            torch.nn.Linear(128, num_input_size)
        )
    def forward(self, x):
        x = self.decoder(x)
        return x 

class Autoencoder(pl.LightningModule):

    def __init__(self,
                 latent_dim: int,
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 num_input_size : int = 53,
                 ):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_size, latent_dim)
        self.decoder = decoder_class(num_input_size, latent_dim)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x = batch # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss.mean())
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss.mean())

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss.mean())






class GenerateCallback(pl.Callback):

    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)



def train_cifar(latent_dim):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"bandits_{latent_dim}"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=500,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    GenerateCallback(get_train_data(8), every_n_epochs=10),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"bandits_{latent_dim}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = Autoencoder(latent_dim=latent_dim)
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result




model_dict = {}
for latent_dim in [64, 128, 256, 384]:
    model_ld, result_ld = train_cifar(latent_dim)
    model_dict[latent_dim] = {"model": model_ld, "result": result_ld}





# class Encoder(nn.Module):
#     def __init__(self, 
#                 num_input_size:int,  
#                 latent_dim:int):
#         super().__init__()

#         self.fc1 = nn.Linear(num_input_size, out_features=64)
#         self.fc2 = nn.Linear(in_features=64, out_features=16)
#         self.fc3 = nn.Linear(in_features=16, latent_dim)

#     def forward(self, x):
#         x = torch.r
#         return self.encoder(x)

# class Decoder(nn.Module):
#     def __init__(self,
#                 num_input_size : int,
#                 latent_dim : int, 
#                 act_fn: object = nn.ReLU):
#         super().__init__()
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, out_features=16),
#             act_fn(),
#             nn.Linear(in_features=16, out_features=64),
#             act_fn(),
#             nn.Linear(in_features=64, num_input_size),
#         )
#     def forward(self, x):
#         x = self.decoder(x)
#         return x 


