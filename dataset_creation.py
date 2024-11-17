from pathlib import Path

import torch
import random
import numpy as np
from torchvision import transforms
import albumentations as A
from torchvision.transforms import functional as F





# Define augmentation pipeline 
transform = A.Compose([
            A.Affine(scale=(0.85, 1.15), rotate=(-45, 45)),  # Zoom and rotate
            A.ElasticTransform()  # Elastic deformations
        ], additional_targets={'mask': 'mask'})

class ProstateDataset(torch.utils.data.Dataset):
    def __init__(self, root, augment_params):
        self.all_files = self.extract_files(root)
        self.augment_params = augment_params

    @staticmethod
    def extract_files(root):
        """
        Extract the paths to all slices given the root path (ends with train or val or test)
        """
        files = []
        for subject in root.glob("*"):   # Iterate over the subjects
            slice_path = subject/"data"  # Get the slices for current subject
            for slice in slice_path.glob("*.npy"):
                files.append(slice)
        return files


    @staticmethod
    def change_img_to_label_path(path):
        """
        Replace data with mask to get the masks
        """
        parts = list(path.parts)
        parts[parts.index("data")] = "masks"
        return Path(*parts)

  #  def augment(self, slice, mask):
        """
        Augments slice and segmentation mask in the exact same way
        Note the manual seed initialization
        """
        ###################IMPORTANT###################
        # Fix for https://discuss.pytorch.org/t/dataloader-workers-generate-the-same-random-augmentations/28830/2
   #     random_seed = torch.randint(0, 1000000, (1,)).item()
    #    imgaug.seed(random_seed)
        #####################################################
     #   mask = SegmentationMapsOnImage(mask, mask.shape)
      #  slice_aug, mask_aug = self.augment_params(image=slice, segmentation_maps=mask)
       # mask_aug = mask_aug.get_arr()
       # return slice_aug, mask_aug
    # Set a random seed for reproducibility
        random_seed = torch.randint(0, 1000000, (1,)).item()
        torch.manual_seed(random_seed)
        
        
        
        # Augmentation function for both image and mask
        
        augmented = self.transform(image=np.array(slice), mask=np.array(mask))
        augmented_image = Image.fromarray(augmented['image'])
        augmented_mask = Image.fromarray(augmented['mask'])
        return augmented_image, augmented_mask

    

    

    def __len__(self):
        """
        Return the length of the dataset (length of all files)
        """
        return len(self.all_files)


    def __getitem__(self, idx):
        """
        Given an index return the (augmented) slice and corresponding mask
        Add another dimension for pytorch
        """
        file_path = self.all_files[idx]
        mask_path = self.change_img_to_label_path(file_path)
        slice = np.load(file_path).astype(np.float32)  # Convert to float for torch
        mask = np.load(mask_path)

        #if self.augment_params:
            #slice, mask = self.augment(slice, mask)
            
        # Note that pytorch expects the input of shape BxCxHxW, where B corresponds to the batch size, C to the channels, H to the height and W to Width.
        # As our data is of shape (HxW) we need to manually add the C axis by using expand_dims.
        # The batch dimension is later added by the dataloader

        return np.expand_dims(slice, 0), np.expand_dims(mask, 0)


#import imgaug.augmenters as iaa

# Create the dataset object
path = Path("Preprocessed/train/")
dataset = ProstateDataset(path, transform)



