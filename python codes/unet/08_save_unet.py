# ============================================================
# IMPORT LIBRARIES
# ============================================================

# PyTorch for saving model
import torch

# File handling
import os

# Import trained model
from create_unet import model


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__)
                    )
                )
            )


# ============================================================
# CREATE MODEL FOLDER
# ============================================================

model_folder = os.path.join(

    BASE_DIR,

    "models",

    "unet"
)


# Create folder if missing
os.makedirs(

    model_folder,

    exist_ok=True
)


# ============================================================
# SAVE MODEL WEIGHTS
# ============================================================

torch.save(

    model.state_dict(),

    os.path.join(

        model_folder,

        "unet_model.pth"

    )
)


print()

print("="*50)

print("U-Net model saved successfully")

print("="*50)