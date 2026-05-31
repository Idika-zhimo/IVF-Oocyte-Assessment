# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os
import cv2

# Augmentation library
import albumentations as A


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
# ORIGINAL DATASET
# ============================================================

image_dir=os.path.join(

    BASE_DIR,
    "datasets",
    "yolo_dataset",
    "images",
    "train"
)


mask_dir=os.path.join(

    BASE_DIR,
    "datasets",
    "yolo_dataset",
    "masks",
    "train"
)


# ============================================================
# OUTPUT FOLDER
# ============================================================

aug_image_dir=os.path.join(

    BASE_DIR,
    "datasets",
    "augmented_dataset",
    "images",
    "train"
)


aug_mask_dir=os.path.join(

    BASE_DIR,
    "datasets",
    "augmented_dataset",
    "masks",
    "train"
)


os.makedirs(

    aug_image_dir,

    exist_ok=True
)

os.makedirs(

    aug_mask_dir,

    exist_ok=True
)


# ============================================================
# AUGMENTATION PIPELINE
# ============================================================

transform=A.Compose([

    A.HorizontalFlip(

        p=0.5
    ),

    A.RandomBrightnessContrast(

        p=0.5
    ),

    A.Rotate(

        limit=20,

        p=0.5
    )

])


print(

    "Augmentation pipeline ready"
)