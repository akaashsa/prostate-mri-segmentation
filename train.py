import torch
import random
from torchvision import transforms
import albumentations as A
from torchvision.transforms import functional as F

import sys
import os
import shutil

sys.path.append('/scratch/ams9696/guided_studies/.local/')
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import wandb


from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
import datetime
from dataset_creation import ProstateDataset
from model import UNet

# now = str(datetime.datetime.now())
# #CHANGE LOG NAME LATER PLEASE
# wandb_logger = WandbLogger(project='prostatemrisegmentation',name='experiment_'+'now')

# # add your batch size to the wandb config
# wandb_logger.experiment.config["batch_size"] = batch_size

# # Create the dataset objects
# train_path = Path("Preprocessed/train/")
# val_path = Path("Preprocessed/val")
# test_path = Path("Preprocessed_test/")

# train_dataset = ProstateDataset(train_path, transform)
# val_dataset = ProstateDataset(val_path, None)
# test_dataset = ProstateDataset(test_path, transform)

# print(f"There are {len(train_dataset)} train images and {len(val_dataset)} val images and {len(test_dataset)} test images")



# batch_size = 8
# num_workers = 4

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


class DiceLoss(torch.nn.Module):
    """
    class to compute the Dice Loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):

        # Flatten label and prediction tensors
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)
        counter = (pred * mask).sum()  # Numerator
        denum = pred.sum() + mask.sum() + 1e-8  # Denominator. Add a small number to prevent NANS
        dice =  (2*counter)/denum
        return 1 - dice


class ProstateSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = UNet(1,1)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = DiceLoss()

    def forward(self, data):
        return torch.sigmoid(self.model(data))

    def training_step(self, batch, batch_idx):
        mri, mask = batch
        mask = mask.float()
        pred = self(mri)

        loss = self.loss_fn(pred, mask)

        self.log("Train Dice", loss)

        if batch_idx % 50 == 0:
            self.log_images(mri.cpu(), pred.cpu(), mask.cpu(), "Train")

        return loss

    def validation_step(self, batch, batch_idx):
        mri, mask = batch
        mask = mask.float()
        pred = self(mri)

        loss = self.loss_fn(pred, mask)

        self.log("Val Dice", loss)

        if batch_idx % 2 == 0:
            self.log_images(mri.cpu(), pred.cpu(), mask.cpu(), "Val")

        return loss

    def test_step(self, batch, batch_idx):
        mri, mask = batch
        mask = mask.float()
        pred = self(mri)

        loss = self.loss_fn(pred, mask)

        self.log("Final Dice", loss)

        if batch_idx % 2 == 0:
            self.log_images(mri.cpu(), pred.cpu(), mask.cpu(), "Test")

        return loss

    def log_images(self, mri, pred, mask, name):

        pred = pred > 0.5

        fig, axis = plt.subplots(1, 2)
        axis[0].imshow(mri[0][0], cmap="bone")
        mask_ = np.ma.masked_where(mask[0][0] == 0, mask[0][0])
        axis[0].imshow(mask_, alpha=0.6)

        axis[1].imshow(mri[0][0], cmap="bone")
        mask_ = np.ma.masked_where(pred[0][0] == 0, pred[0][0])
        axis[1].imshow(mask_, alpha=0.6)

        wandb_image = []
        wandb_image.append(wandb.Image(fig))
        self.logger.experiment.log({'image_logged': wandb_image})

    def configure_optimizers(self):
        return [self.optimizer]
    
    
#     # Instanciate the model and set the random seed
# torch.manual_seed(0)
# model = ProstateSegmentation()

# # Create the checkpoint callback
# checkpoint_callback_val = ModelCheckpoint(
#     monitor='Val Dice',
#     save_top_k=20,
#     mode='min',
#     dirpath = './guided_studies/mri_image_segmentations/val/')

# checkpoint_callback_train = ModelCheckpoint(
#     monitor='Train Dice',
#     save_top_k=20,
#     mode='min',
#     dirpath = './guided_studies/mri_image_segmentations/train/')


# torch.cuda.is_available()

# # Specify the number of GPUs. Use 'cpu' for CPU training, 'gpu' for GPU training
# gpus = 1  # You can change this to 0 for CPU, or increase for multiple GPUs

# # Create the trainer
# trainer = pl.Trainer(
#     accelerator="gpu" if gpus > 0 else "cpu",  # Use "gpu" if gpus is greater than 0, otherwise "cpu"
#     devices=gpus if gpus > 0 else None,       # Specify the number of devices (GPUs) or None for CPU
#     logger=wandb_logger,
#     log_every_n_steps=1,
#     callbacks=[checkpoint_callback_val,checkpoint_callback_train],          # Wrap the callback in a list
#     max_epochs=75
# )

# trainer.fit(model, train_loader, val_loader)


# Path to the checkpoints folder

# def resume_training(checkpoint_folder,pl):
    
#     checkpoint_folder_val = checkpoint_folder[0]
#     checkpoint_folder_train =checkpoint_folder[1]
#     print(len(glob.glob(os.path.join(checkpoint_folder_val))))
#     latest_checkpoint_val= max(glob.glob(os.path.join(checkpoint_folder_val, "*.ckpt")), key=os.path.getctime)
    
#     print(glob.glob(os.path.join(checkpoint_folder_train)))
#     latest_checkpoint_train = max(glob.glob(os.path.join(checkpoint_folder_train, "*.ckpt")), key=os.path.getctime)
#     print(f"Latest checkpoint found at: {latest_checkpoint_val} and {latest_checkpoint_train} ")
#     latest_checkpoint = "guided_studies/mri_image_segmentations/val/"
    
#     trainer = pl.Trainer(
#         accelerator="gpu" if gpus > 0 else "cpu",
#         devices=gpus if gpus > 0 else None,
#         logger=wandb_logger,
#         log_every_n_steps=1,
#         callbacks=[checkpoint_callback_val,checkpoint_callback_train],
#         max_epochs=75  # Path to the checkpoint to resume from
#     )

#     trainer.fit(model, train_loader, val_loader, ckpt_path=latest_checkpoint_val)

