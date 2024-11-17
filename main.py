#!/bin/env python
import torch    
import sys
import os
import shutil

sys.path.append('/scratch/ams9696/guided_studies/.local/')
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import argparse
import torch
import random
from torchvision import transforms
import albumentations as A
import pytorch_lightning as pl
from pathlib import Path
from train import ProstateSegmentation, DiceLoss
from dataset_creation import ProstateDataset
from model import UNet
from validation import get_val_dice_score,get_test_dice_score

from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import datetime

# Main script to run training, validation, and testing
def main(args):
    # Initialize wandb logger if required
    wandb_logger = None
    
    if args.use_wandb:
        from pytorch_lightning.loggers import WandbLogger
        now = str(datetime.datetime.now())
        # #CHANGE LOG NAME LATER PLEASE
        wandb_logger = WandbLogger(project='prostatemrisegmentation',name='prostatesegexperiment_'+now,log_model='all',resume="allow")

    # Define augmentation pipeline 
    transform = A.Compose([
            A.Affine(scale=(0.85, 1.15), rotate=(-45, 45)),  # Zoom and rotate
            A.ElasticTransform()  # Elastic deformations
        ], additional_targets={'mask': 'mask'})

    # Initialize datasets

    print("Args ",args)
    print("Train path",args.train_data_path )
    from pathlib import Path

    train_path = Path(args.train_data_path)
    if not train_path.exists() or len(list(train_path.glob("*"))) == 0:
        print("Warning: The train dataset directory is empty or doesn't exist.")
    else :
        print(len(list(train_path.glob("*"))))

    train_dataset = ProstateDataset(Path(args.train_data_path),transform)
    val_dataset = ProstateDataset(Path(args.val_data_path),None)
    test_dataset = ProstateDataset(Path(args.test_data_path),None)

    # sweep_config = {
    #         'method': 'random'
    #         }
        
    # metric = {
    #         'name': 'loss',
    #         'goal': 'minimize'   
    #         }
        
    # sweep_config['metric'] = metric
    # parameters_dict = {
    #         'optimizer': {
    #             'values': ['adam', 'sgd']
    #             },
    #         'fc_layer_size': {
    #             'values': [128, 256, 512]
    #             },
    #         'max_epochs':{
    #             'distribution': 'uniform',
    #             'min': 5,
    #             'max': 25
    #         },
    #         'batch_size': {
    #             # integers between 32 and 256
    #             # with evenly-distributed logarithms 
    #             'distribution': 'q_log_uniform_values',
    #             'q': 8,
    #             'min': 8,
    #             'max': 32,
    #           },
    #         'dropout': {
    #               'values': [0.3, 0.4, 0.5]
    #             },
    #         }
        
    # sweep_config['parameters'] = parameters_dict
        # add your batch size to the wandb config
    # wandb_logger.experiment.config["batch_size"] = batch_size
    # sweep_id = wandb.sweep(sweep_config, project='prostatemrisegmentation')


    
        
    
    # Model and Training Setup
    # def train(config=None):
    #     with wandb.init(config=config):
    #         # Access hyperparameters
    #         config = wandb.config
    
    #         # Initialize dataloaders
    #         # Initialize dataloaders
           
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        
    print(f"There are {len(train_dataset)} train images and {len(val_dataset)} val images and {len(test_dataset)} test images")
    
    
            # Model setup
    torch.manual_seed(0)
    model = ProstateSegmentation()
    
              # Create the checkpoint callback
    checkpoint_callback_val = ModelCheckpoint(
                monitor='Val Dice',
                every_n_epochs=10,
                save_last=True,
                mode='min',
                dirpath = './guided_studies/latest/mri_image_segmentations/val/')
        
    checkpoint_callback_train = ModelCheckpoint(
                monitor='Train Dice',
                every_n_epochs=10,
                save_last=True,
                mode='min',
                dirpath = './guided_studies/latest/mri_image_segmentations/train/')
     
            # Trainer
             # Create the trainer
    gpus = 1  # You can change this to 0 for CPU, or increase for multiple GPUs
    trainer = pl.Trainer(
                accelerator="gpu" if gpus > 0 else "cpu",  # Use "gpu" if gpus is greater than 0, otherwise "cpu"
                devices=gpus if gpus > 0 else None,       # Specify the number of devices (GPUs) or None for CPU
                logger=wandb_logger,
                log_every_n_steps=1,
                callbacks=[checkpoint_callback_val,checkpoint_callback_train],          # Wrap the callback in a list
                max_epochs=args.epochs
            )
        
                # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    val_dice_score_my_method = get_val_dice_score(val_datasest,model)
    test_dice_score_my_method = get_test_dice_score(test_dataset,model)
    print(f"Dice Scores my method : Val : {val_dice_score}, Test : {test_dice_score}")
        
            # Validate final model on validation set
    val_dice_score = trainer.validate(model, val_loader)
    print(f"Validation Dice Score : {val_dice_score}")
    
            # Evaluate on the test set
    test_results = trainer.test(model, test_loader)
    print(f"Test Results: {test_results}")
       
    
  
   

    # # Train and validate
    
    # trainer.fit(model, train_loader, val_loader)

    # checkpoint_folder_val = "guided_studies/mri_image_segmentations/val/"
    # checkpoint_folder_train ="guided_studies/mri_image_segmentations/train/"
    # checkpoint_folder = [checkpoint_folder_val,checkpoint_folder_train]

  

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, validate, and test prostate MRI segmentation model")
    parser.add_argument('--project_name', type=str, default='prostatemrisegmentation', help='W&B project name')
    parser.add_argument('--train_data_path', type=str, default='/scratch/ams9696/Preprocessed/train/', help='Path to training data')
    parser.add_argument('--val_data_path', type=str, default='/scratch/ams9696/Preprocessed/val/', help='Path to validation data')
    parser.add_argument('--test_data_path', type=str, default='/scratch/ams9696/Preprocessed_test/', help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Num workers')
    parser.add_argument('--epochs', type=int, default=75, help='Number of training epochs')
    parser.add_argument('--use_wandb', default = True,action='store_true', help='Use WandB for logging')    
    args = parser.parse_args()
    # sweep_id = wandb.sweep(sweep_config, project=args.project_name)

    # # Start the sweep
    # wandb.agent(sweep_id, train, count=5)
    
    
    main(args)
