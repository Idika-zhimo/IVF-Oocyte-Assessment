# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


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
# CREATE U-NET MODEL
# ============================================================

model=smp.Unet(

    encoder_name="resnet34",

    encoder_weights=None,

    in_channels=3,

    classes=4
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model.load_state_dict(

    torch.load(

        os.path.join(

            BASE_DIR,

            "models",

            "unet",

            "unet_model.pth"

        ),

        map_location="cpu"
    )
)


model.eval()


# ============================================================
# TEST IMAGE FOLDER
# ============================================================

test_folder=os.path.join(

    BASE_DIR,

    "datasets",

    "yolo_dataset",

    "images",

    "test"
)


# ============================================================
# OUTPUT FOLDER
# ============================================================

output_folder=os.path.join(

    BASE_DIR,

    "results",

    "unet_predictions"
)


os.makedirs(

    output_folder,

    exist_ok=True
)


# ============================================================
# PREDICT IMAGES
# ============================================================

for image_name in os.listdir(test_folder):


    image_path=os.path.join(

        test_folder,

        image_name
    )


    image=cv2.imread(image_path)

    image=cv2.resize(

        image,

        (256,256)
    )


    image=image/255.0


    image=np.transpose(

        image,

        (2,0,1)
    )


    image=np.expand_dims(

        image,

        axis=0
    )


    image=torch.tensor(

        image,

        dtype=torch.float32
    )


    with torch.no_grad():

        prediction=model(

            image
        )


    prediction=torch.argmax(

        prediction,

        dim=1
    )


    prediction=prediction.squeeze().numpy()


    save_path=os.path.join(

        output_folder,

        image_name
    )


    cv2.imwrite(

        save_path,

        prediction*50
    )


    print(

        "Saved:",

        image_name
    )


print()

print("U-Net testing completed")