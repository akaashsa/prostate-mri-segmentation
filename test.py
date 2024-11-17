import os
import glob
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
from train import DiceLoss

# Path to the checkpoints folder
checkpoint_folder_val = "guided_studies/mri_image_segmentations/val/"
checkpoint_folder_train = "guided_studies/mri_image_segmentations/train/"
print(glob.glob(os.path.join(checkpoint_folder_val)))
latest_checkpoint_val= max(glob.glob(os.path.join(checkpoint_folder_val, "*.ckpt")), key=os.path.getctime)

print(glob.glob(os.path.join(checkpoint_folder_train)))
latest_checkpoint_train = max(glob.glob(os.path.join(checkpoint_folder_train, "*.ckpt")), key=os.path.getctime)
print(f"Latest checkpoint found at: {latest_checkpoint_val} and {latest_checkpoint_train} ")
latest_checkpoint = "guided_studies/mri_image_segmentations/val/a"

trainer = pl.Trainer(
    accelerator="gpu" if gpus > 0 else "cpu",
    devices=gpus if gpus > 0 else None,
    logger=wandb_logger,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback_val,checkpoint_callback_train],
    max_epochs=75  # Path to the checkpoint to resume from
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.eval();
model.to(device);

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the model from the latest checkpoint
model = ProstateSegmentation().to(device);
#model = ProstateSegmentation.load_from_checkpoint(latest_checkpoint)
model.eval();
model.to(device);

preds = []
labels = []

for slice, label in tqdm(val_dataset):
    slice = torch.tensor(slice).to(device).unsqueeze(0)
    with torch.no_grad():
        pred = model(slice)
    preds.append(pred.cpu().numpy())
    labels.append(label)
    
preds = np.array(preds)
labels = np.array(labels)


1-model.loss_fn(torch.from_numpy(preds), torch.from_numpy(labels))  # two possibilities

dice_score = 1-DiceLoss()(torch.from_numpy(preds), torch.from_numpy(labels).unsqueeze(0).float())
print(f"The Val Dice Score is: {dice_score}")