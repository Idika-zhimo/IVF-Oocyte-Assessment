# ============================================================
# IMPORT YOLO
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
# YAML LOCATION
# ============================================================

yaml_path = os.path.join(

    BASE_DIR,

    "dataset.yaml"

)


# ============================================================
# RESULTS LOCATION
# ============================================================

results_folder = os.path.join(

    BASE_DIR,

    "runs"

)


# ============================================================
# LOAD PRETRAINED YOLOv8 SEGMENTATION MODEL
# ============================================================

# n = nano version

# seg = segmentation model

model = YOLO(

    "yolov8n-seg.pt"

)


# ============================================================
# START TRAINING
# ============================================================

model.train(

    data=yaml_path,

    task="segment",

    epochs=100,

    imgsz=640,

    batch=16,

    project=results_folder,

    name="yolov8_segmentation"

)


print()

print("="*50)

print("Training Completed")

print("="*50)