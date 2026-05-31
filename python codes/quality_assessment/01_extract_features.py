# ======================================================
# IMPORT LIBRARIES
# ======================================================

import os
import cv2
import numpy as np
import pandas as pd


# ======================================================
# MASK FOLDER LOCATION
# ======================================================

mask_folder = r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\datasets\augmented_dataset\masks\train"


# ======================================================
# GET ALL MASK FILES
# ======================================================

mask_files = os.listdir(mask_folder)

features=[]


# ======================================================
# PROCESS EACH MASK
# ======================================================

for file in mask_files:

    path=os.path.join(
        mask_folder,
        file
    )

    mask=cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        continue


    # ==================================================
    # EXTRACT AREAS
    # ==================================================

    cytoplasm_area=np.sum(mask==1)

    polar_body_area=np.sum(mask==2)

    zona_area=np.sum(mask==3)

    total_area=(
        cytoplasm_area+
        polar_body_area+
        zona_area
    )


    # ==================================================
    # POLAR BODY DETECTION
    # ==================================================

    polar_present=0

    if polar_body_area>0:

        polar_present=1


    # ==================================================
    # POLAR BODY RATIO
    # ==================================================

    polar_ratio=0

    if cytoplasm_area>0:

        polar_ratio=(

            polar_body_area/

            cytoplasm_area
        )


    # ==================================================
    # ZONA THICKNESS RATIO
    # ==================================================

    zona_ratio=0

    if cytoplasm_area>0:

        zona_ratio=(

            zona_area/

            cytoplasm_area
        )


    # ==================================================
    # SHAPE / CIRCULARITY
    # ==================================================

    binary=(mask>0).astype(np.uint8)

    contours,_=cv2.findContours(

        binary,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )


    circularity=0


    if len(contours)>0:

        largest=max(

            contours,

            key=cv2.contourArea
        )


        area=cv2.contourArea(
            largest
        )


        perimeter=cv2.arcLength(

            largest,

            True
        )


        if perimeter>0:

            circularity=(

                4*
                np.pi*
                area

            )/(perimeter**2)


    # ==================================================
    # SAVE FEATURES
    # ==================================================

    features.append([

        file,

        cytoplasm_area,

        polar_body_area,

        polar_present,

        polar_ratio,

        zona_area,

        zona_ratio,

        total_area,

        circularity

    ])


# ======================================================
# CREATE DATAFRAME
# ======================================================

df=pd.DataFrame(

features,

columns=[

'image_name',
'cytoplasm_area',
'polar_body_area',
'polar_present',
'polar_ratio',
'zona_area',
'zona_ratio',
'total_area',
'circularity'

]

)


# ======================================================
# SAVE CSV
# ======================================================

save_path=r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\quality_features.csv"

df.to_csv(

save_path,

index=False
)


# ======================================================
# PRINT RESULTS
# ======================================================

print("="*50)

print("Feature extraction completed")

print("="*50)

print()

print("Saved at:")

print(save_path)

print()

print(df.head())