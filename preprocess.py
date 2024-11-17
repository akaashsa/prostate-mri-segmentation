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

import posixpath
# Step 1: Add custom directory to sys.path if needed

# Step 2: Install required packages (preferably without --target)
# !pip install --user FuzzyTM>=0.4.0 --upgrade
# !pip install --user numpy --upgrade
# !pip install --user tensorboardX --upgrade
# !pip install --user imgaug

# Step 3: Restart the kernel to ensure packages are loaded
import IPython
app = IPython.Application.instance()

# Step 4: Import packages
import matplotlib.pyplot as plt
#import imgaug.augmenters as iaa
#from imgaug.augmentables.segmaps import SegmentationMapsOnImage

root = Path("./guided_studies/final_data/train/")
label = Path("/Case00_site1_segmentation.nii.gz/")


# Identify sites which have prostate
# Out

# def convert_to_lower(output_directory):
#     """
#     Replaces uppercase "Segmentation" labels with "segmentation" in filenames
#     and copies them to the output directory.
#     """
    
#     updated_files = []
#     output_directory = Path(output_directory)
#     for file_path in output_directory.iterdir():
#         path = Path(file_path)
#         filename = path.name  # Get the file name

#         # Check if the filename ends with either uppercase or lowercase "Segmentation"
#         if filename.endswith('_Segmentation.nii.gz') or filename.endswith('_segmentation.nii.gz'):
#             # Construct the new file name
#             case_name, extension = filename.split('_', 1)
#             new_file_name = f"{case_name}_{extension.lower()}"
            
#             # Define source and destination paths
#             src_path = str(path)
#             dest_path = os.path.join(output_directory, new_file_name)
            
#             # Copy file to new destination with the updated name
#             shutil.move(src_path, dest_path)
      
def remove_directory_and_check(directory_path):

    try:
        # Remove the directory and all its contents
        shutil.rmtree(directory_path, ignore_errors=True)
    
        # Check if directory exists and count files
        if os.path.exists(directory_path):
            # If the directory still exists, count files
            num_files = sum([len(files) for _, _, files in os.walk(directory_path)])
        else:
            # Directory doesn't exist, so no files are left
            num_files = 0
            
        print(f"{directory_path} cleared successfully, currently there are {num_files} in this directory.")
        return num_files

    except OSError as e:
        print(f"Error: {e.strerror}")
   
            
def change_img_to_label_path(path):
    """
    Replaces imagesTr with labelsTr
    """
    filename = path.name  # Gets the last part of the path (the file name)
    
    # turn all files to lower case
    if not filename.endswith('_segmentation.nii.gz'):

        filename = filename.replace('.nii.gz', '_segmentation.nii.gz')

    else:

        filename = filename



    # Create a new path with the updated file name
    new_path = path.with_name(filename)
    return new_path



sample_path = (list(root.glob("./Case*")))[10]  # Choose a subject
print(sample_path)
sample_path_label = change_img_to_label_path(sample_path)


# loading image using nib.load
data = nib.load(sample_path)
label = nib.load(sample_path_label)
# extract image from nifti file using its intrinsic fdata property
mri = data.get_fdata()
#extract labelled/segmented image from nifti file
mask = label.get_fdata().astype(np.uint8) 


# Helper functions for normalization and standardization
def normalize(full_volume):
    """
    Z-Normalization of the whole subject
    """
    mu = full_volume.mean()
    std = np.std(full_volume)
    normalized = (full_volume - mu) / std
    return normalized

def standardize(normalized_data):
    """
    Standardize the normalized data into the 0-1 range
    """
    standardized_data = (normalized_data - normalized_data.min()) / (normalized_data.max() - normalized_data.min())
    return standardized_data


all_files = list(root.glob("./Case*"))  # Get all subjects

save_root = Path("Preprocessed") 

# clean out the prexisting train and val slices from prev run
directory_path = "Preprocessed"
remove_directory_and_check(directory_path)


#use uniform sizes for all the images, check the largest size of immages and just append 0s to it
for counter, path_to_mri_data in enumerate(tqdm(all_files)):

    path_to_label = change_img_to_label_path(path_to_mri_data)

    mri = nib.load(path_to_mri_data)
    assert nib.aff2axcodes(mri.affine) == ("L", "P", "S")
    mri_data = mri.get_fdata()
    label_data = nib.load(path_to_label).get_fdata().astype(np.uint8)

    # Crop volume and label mask. Reduce 64 px from top and 64 px from bottom (128 px in total).
    # Addtionally crop front and back with same size. Dont crop viewing axis
    # Cropping is done in accordance to the paper to get 256x256 size
    mri_data = mri_data[64:-64, 64:-64]
    label_data = label_data[64:-64, 64:-64]
    # Normalize and standardize the images
    normalized_mri_data = normalize(mri_data)
    standardized_mri_data = standardize(normalized_mri_data)

    # Check if train or val data and create corresponding path
    # Training only using one site for 156 out of 206 patients
    if counter < 156:
        current_path = save_root/"train"/str(counter) # training will contain all the sites other site a
    else:
        current_path = save_root/"val"/str(counter) # validation will just contain site a

    # Loop over the slices in the full volume and store the images and labels in the data/masks directory
    for i in range(standardized_mri_data.shape[-1]):
        slice = standardized_mri_data[:,:,i]
        mask = label_data[:,:,i]
        slice_path = current_path/"data"
        mask_path = current_path/"masks"
        slice_path.mkdir(parents=True, exist_ok=True)
        mask_path.mkdir(parents=True, exist_ok=True)

        np.save(slice_path/str(i), slice)

        np.save(mask_path/str(i), mask)



#Processing test data as well
root_test = Path("./guided_studies/final_data/test/")
all_files_test = list(root_test.glob("./Case*"))  # Get all subjects

# clean out the prexisting test slices from prev run
directory_path_test = "Preprocessed_test"
remove_directory_and_check(directory_path_test)


save_root = Path("Preprocessed_test") 
#use uniform sizes for all the images, check the largest size of immages and just append 0s to it
for counter, path_to_mri_data in enumerate(tqdm(all_files_test)):

    path_to_label = change_img_to_label_path(path_to_mri_data)

    mri = nib.load(path_to_mri_data)
    assert nib.aff2axcodes(mri.affine) == ("L", "P", "S")
    mri_data = mri.get_fdata()
    label_data = nib.load(path_to_label).get_fdata().astype(np.uint8)

    # Crop volume and label mask. Reduce 64 px from top and 64 px from bottom (128 px in total).
    # Addtionally crop front and back with same size. Dont crop viewing axis
    # Cropping is done in accordance to the paper to get 256x256 size
    mri_data = mri_data[64:-64, 64:-64]
    label_data = label_data[64:-64, 64:-64]
    # Normalize and standardize the images
    normalized_mri_data = normalize(mri_data)
    standardized_mri_data = standardize(normalized_mri_data)

    current_path = save_root/str(counter)
   
    # Loop over the slices in the full volume and store the images and labels in the data/masks directory
    for i in range(standardized_mri_data.shape[-1]):
        slice = standardized_mri_data[:,:,i]
        mask = label_data[:,:,i]
        slice_path = current_path/"data"
        mask_path = current_path/"masks"
        slice_path.mkdir(parents=True, exist_ok=True)
        mask_path.mkdir(parents=True, exist_ok=True)

        np.save(slice_path/str(i), slice)

        np.save(mask_path/str(i), mask)


