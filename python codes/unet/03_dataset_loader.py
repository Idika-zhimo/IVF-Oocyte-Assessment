# ============================================================
# IMPORT LIBRARIES
# ============================================================

# File handling
import os

# Image processing
import cv2

# Numerical operations
import numpy as np

# PyTorch Dataset and DataLoader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# ============================================================
# CUSTOM DATASET CLASS
# ============================================================

class OocyteDataset(Dataset):


    # Constructor runs once
    def __init__(

        self,

        image_dir,

        mask_dir

    ):


        # Store image folder path

        self.image_dir=image_dir


        # Store mask folder path

        self.mask_dir=mask_dir


        # Read all image names

        self.images=os.listdir(

            image_dir
        )


    # Return total image count

    def __len__(self):

        return len(

            self.images
        )


    # Load one image-mask pair

    def __getitem__(self,index):


        image_name=self.images[index]


        image_path=os.path.join(

            self.image_dir,

            image_name
        )


        image=cv2.imread(

            image_path
        )


        # Convert image name to mask name

        mask_name=(

            os.path.splitext(
                image_name
            )[0]

            +".png"
        )


        mask_path=os.path.join(

            self.mask_dir,

            mask_name
        )


        mask=cv2.imread(

            mask_path,

            0
        )


        # Normalize image

        image=image/255.0


        return image,mask


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR=os.path.dirname(
            os.path.dirname(
            os.path.dirname(
            os.path.abspath(__file__)
            ))
         )


# ============================================================
# TRAIN FOLDERS
# ============================================================

train_image_dir=os.path.join(

    BASE_DIR,

    "datasets",

    "augmented_dataset",

    "images",

    "train"
)


train_mask_dir=os.path.join(

    BASE_DIR,

    "datasets",

    "augmented_dataset",

    "masks",

    "train"
)


# ============================================================
# CREATE DATASET
# ============================================================

train_dataset=OocyteDataset(

    train_image_dir,

    train_mask_dir
)


# ============================================================
# CREATE DATALOADER
# ============================================================

train_loader=DataLoader(

    train_dataset,

    batch_size=4,

    shuffle=True
)


print(

    "Training Samples:",

    len(train_dataset)
)


print(

    "Total batches:",

    len(train_loader)
)