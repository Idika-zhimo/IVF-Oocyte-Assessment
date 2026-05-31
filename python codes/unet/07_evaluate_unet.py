# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os
import cv2
import numpy as np


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
# FOLDERS
# ============================================================

ground_truth_folder=os.path.join(

    BASE_DIR,

    "datasets",

    "yolo_dataset",

    "masks",

    "test"
)


prediction_folder=os.path.join(

    BASE_DIR,

    "results",

    "unet_predictions"
)


# ============================================================
# VARIABLES
# ============================================================

iou_scores=[]

dice_scores=[]


# ============================================================
# LOOP THROUGH MASKS
# ============================================================

for file_name in os.listdir(ground_truth_folder):


    gt_path=os.path.join(

        ground_truth_folder,

        file_name
    )


    pred_path=os.path.join(

        prediction_folder,

        file_name
    )


    if not os.path.exists(pred_path):

        continue


    gt=cv2.imread(

        gt_path,

        0
    )


    pred=cv2.imread(

        pred_path,

        0
    )


    gt=cv2.resize(

        gt,

        (256,256)
    )


    pred=cv2.resize(

        pred,

        (256,256)
    )


    # Convert to binary masks

    gt=(gt>0).astype(np.uint8)

    pred=(pred>0).astype(np.uint8)


    # Intersection

    intersection=np.logical_and(

        gt,

        pred
    )


    # Union

    union=np.logical_or(

        gt,

        pred
    )


    # IoU

    iou=np.sum(intersection)/(

            np.sum(union)

            +1e-6
    )


    # Dice score

    dice=(

            2*np.sum(intersection)

            /

            (

                np.sum(gt)

                +

                np.sum(pred)

                +1e-6
            )
    )


    iou_scores.append(iou)

    dice_scores.append(dice)


# ============================================================
# PRINT FINAL SCORES
# ============================================================

print()

print("="*50)

print("U-Net Evaluation Results")

print("="*50)

print()

print(

    "Average IoU:",

    np.mean(iou_scores)
)

print()

print(

    "Average Dice Score:",

    np.mean(dice_scores)
)

print()

print("="*50)