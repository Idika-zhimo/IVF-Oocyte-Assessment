# ============================================================
# IMPORT LIBRARIES
# ============================================================

from ultralytics import YOLO

import os


# ============================================================
# FIND PROJECT ROOT
# ============================================================

BASE_DIR = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__)
                    )
                )
            )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model_path = os.path.join(

    BASE_DIR,

    "models",

    "yolov8",

    "best.pt"

)


model = YOLO(

    model_path

)


# ============================================================
# TEST IMAGE LOCATION
# ============================================================

test_folder = os.path.join(

    BASE_DIR,

    "datasets",

    "yolo_dataset",

    "images",

    "test"

)


# ============================================================
# RUN PREDICTION
# ============================================================

results = model.predict(

    source=test_folder,

    conf=0.25,

    save=True,

    show=False

)


print()

print("="*50)

print("Prediction completed")

print("="*50)