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
from train import DiceLoss, ProstateSegmentation



def get_val_dice_score(val_dataset,model):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model from the latest checkpoint
    model = model.to(device);
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

    return dice_score



def get_test_dice_score(test_dataset,model):
    preds_test = []
    labels_test = []
    
    #Run on test dataset
    for slice, label in tqdm(test_dataset):
        slice = torch.tensor(slice).to(device).unsqueeze(0)
        with torch.no_grad():
            pred = model(slice)
        preds_test.append(pred.cpu().numpy())
        labels_test.append(label)
    
    preds_test = np.array(preds_test)
    labels_test = np.array(labels_test)
    
    test_dice_score = 1-DiceLoss()(torch.from_numpy(preds_test), torch.from_numpy(labels_test).unsqueeze(0).float())
    print(f"The Test Dice Score is: {test_dice_score}")
