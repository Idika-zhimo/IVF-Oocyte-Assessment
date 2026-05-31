# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

# os
# Used for working with folders and file paths
# Example:
# joining folder path + filename

import os


# cv2 (OpenCV)
# Used for reading images and image processing operations

import cv2


# numpy
# Used for mathematical calculations on images

import numpy as np


# pandas
# Used for creating tables (DataFrames)
# and saving CSV files

import pandas as pd



# ============================================================
# DEFINE MASK FOLDER LOCATION
# ============================================================

# r before the string means:
# "raw string"

# This prevents Windows "\" problems

# Change this path if your folder location changes

mask_folder = r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\augmented_dataset\masks\train"



# ============================================================
# GET ALL FILENAMES INSIDE FOLDER
# ============================================================

# os.listdir()
# returns:

# [
#   "img1.png",
#   "img2.png",
#   "img3.png"
# ]

mask_files = os.listdir(mask_folder)



# ============================================================
# CREATE EMPTY LIST
# ============================================================

# This list will store extracted features

features = []



# ============================================================
# LOOP THROUGH EVERY MASK IMAGE
# ============================================================

# Example:

# First iteration:
# file="img1.png"

# Second iteration:
# file="img2.png"

for file in mask_files:



    # ========================================================
    # CREATE COMPLETE IMAGE PATH
    # ========================================================

    # Combines:

    # folder path
    # +
    # image filename

    # Example:

    # C:/folder/img1.png

    mask_path = os.path.join(

        mask_folder,

        file
    )



    # ========================================================
    # LOAD MASK IMAGE
    # ========================================================

    # cv2.imread() loads image

    # cv2.IMREAD_GRAYSCALE means:

    # convert image into one channel

    # pixel values become:

    # 0
    # 1
    # 2
    # 3

    # instead of RGB colors

    mask = cv2.imread(

        mask_path,

        cv2.IMREAD_GRAYSCALE
    )



    # ========================================================
    # SKIP BROKEN IMAGES
    # ========================================================

    # If image cannot load

    # move to next image

    if mask is None:

        continue



    # ========================================================
    # CALCULATE AREA OF EACH CLASS
    # ========================================================

    # mask==1 creates:

    # True False True True ...

    # np.sum() counts total True values

    # This becomes number of pixels



    # Cytoplasm pixels

    cytoplasm_area = np.sum(

        mask == 1
    )



    # Polar body pixels

    polar_body_area = np.sum(

        mask == 2
    )



    # Zona pellucida pixels

    zona_area = np.sum(

        mask == 3
    )



    # ========================================================
    # CALCULATE TOTAL SEGMENTED AREA
    # ========================================================

    # Add all biological structures together

    total_area = (

        cytoplasm_area +

        polar_body_area +

        zona_area
    )



    # ========================================================
    # CREATE BINARY MASK
    # ========================================================

    # Convert:

    # 0,1,2,3

    # into:

    # 0,1

    # because contour detection
    # works better with binary images

    binary = (

        mask > 0

    ).astype(

        np.uint8
    )



    # ========================================================
    # FIND OBJECT BOUNDARIES
    # ========================================================

    # findContours()

    # detects outer boundary

    contours, _ = cv2.findContours(

        binary,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )



    # ========================================================
    # INITIALIZE CIRCULARITY
    # ========================================================

    circularity = 0



    # ========================================================
    # CHECK IF CONTOURS EXIST
    # ========================================================

    if len(contours) > 0:



        # ====================================================
        # FIND LARGEST CONTOUR
        # ====================================================

        # max() selects contour with biggest area

        largest = max(

            contours,

            key=cv2.contourArea
        )



        # ====================================================
        # CALCULATE AREA
        # ====================================================

        area = cv2.contourArea(

            largest
        )



        # ====================================================
        # CALCULATE PERIMETER
        # ====================================================

        perimeter = cv2.arcLength(

            largest,

            True
        )



        # ====================================================
        # AVOID DIVIDE BY ZERO ERROR
        # ====================================================

        if perimeter > 0:



            # ================================================
            # CALCULATE CIRCULARITY
            # ================================================

            # Formula:

            # 4πA/P²

            # Near 1:

            # perfect circle

            # Smaller values:

            # irregular shape

            circularity = (

                4 *

                np.pi *

                area

            ) / (

                perimeter**2
            )



    # ========================================================
    # STORE FEATURES
    # ========================================================

    # Save all extracted values

    features.append([

        file,

        cytoplasm_area,

        polar_body_area,

        zona_area,

        total_area,

        circularity
    ])



# ============================================================
# CREATE TABLE
# ============================================================

# Convert list → dataframe

df = pd.DataFrame(

    features,

    columns=[

        "image_name",

        "cytoplasm_area",

        "polar_body_area",

        "zona_area",

        "total_area",

        "circularity"
    ]
)



# ============================================================
# DEFINE SAVE LOCATION
# ============================================================

save_path = r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\features.csv"



# ============================================================
# SAVE TABLE AS CSV
# ============================================================

df.to_csv(

    save_path,

    index=False
)



# ============================================================
# PRINT RESULTS
# ============================================================

print("="*50)

print("Feature extraction completed")

print("="*50)

print()

print("Saved file location:")

print(save_path)

print()

print("First 5 rows:")

print(df.head())