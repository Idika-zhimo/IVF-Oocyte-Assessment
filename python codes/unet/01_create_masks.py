# ============================================================
# IMPORT LIBRARIES
# ============================================================

# Helps work with folders/files
import os

# Image processing library
import cv2

# Numerical calculations
import numpy as np


# ============================================================
# PROJECT ROOT FOLDER
# ============================================================

BASE_DIR=os.path.dirname(
            os.path.dirname(
            os.path.dirname(
            os.path.abspath(__file__)
            ))
         )


# ============================================================
# DATASET LOCATION
# ============================================================

dataset_path=os.path.join(

    BASE_DIR,
    "datasets",
    "yolo_dataset"
)


# ============================================================
# TRAIN/VAL/TEST FOLDERS
# ============================================================

splits=["train","val","test"]


# ============================================================
# LOOP THROUGH SPLITS
# ============================================================

for split in splits:

    image_dir=os.path.join(
        dataset_path,
        "images",
        split
    )

    label_dir=os.path.join(
        dataset_path,
        "labels",
        split
    )

    mask_dir=os.path.join(
        dataset_path,
        "masks",
        split
    )


    # Create folder if missing
    os.makedirs(

        mask_dir,

        exist_ok=True
    )


    # Read image names

    for image_name in os.listdir(image_dir):


        # Create image path

        image_path=os.path.join(

            image_dir,

            image_name
        )


        # Read image

        image=cv2.imread(

            image_path
        )


        if image is None:

            continue


        # Get image dimensions

        height,width=image.shape[:2]


        # Create empty mask

        mask=np.zeros(

            (height,width),

            dtype=np.uint8
        )


        # Convert image name to label name

        label_name=(

            os.path.splitext(image_name)[0]

            +".txt"

        )


        label_path=os.path.join(

            label_dir,

            label_name
        )


        if not os.path.exists(label_path):

            continue


        with open(label_path) as file:

            lines=file.readlines()


        # Read polygon labels

        for line in lines:


            values=line.strip().split()


            class_id=int(

                values[0]
            )


            polygon=np.array(

                values[1:],

                dtype=np.float32
            )


            polygon=polygon.reshape(

                -1,
                2
            )


            # Convert normalized coordinates

            polygon[:,0]*=width

            polygon[:,1]*=height


            polygon=polygon.astype(

                np.int32
            )


            # Draw object

            cv2.fillPoly(

                mask,

                [polygon],

                color=class_id+1
            )


        # Save mask

        mask_name=(

            os.path.splitext(image_name)[0]

            +".png"
        )


        cv2.imwrite(

            os.path.join(
                mask_dir,
                mask_name
            ),

            mask
        )


print(

    "Mask generation completed"
)