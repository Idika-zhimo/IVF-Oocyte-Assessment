# ============================================================
# IMPORT LIBRARIES
# ============================================================

import torch

import segmentation_models_pytorch as smp


# ============================================================
# SELECT CPU OR GPU
# ============================================================

device=torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)


# ============================================================
# CREATE U-NET MODEL
# ============================================================

model=smp.Unet(


    # Feature extractor

    encoder_name="resnet34",


    # Transfer learning

    encoder_weights="imagenet",


    # RGB channels

    in_channels=3,


    # Classes

    classes=4

)


# Send model to GPU/CPU

model=model.to(

    device
)


print(

    "Model running on:",

    device
)