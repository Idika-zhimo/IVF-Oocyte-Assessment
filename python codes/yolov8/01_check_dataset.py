# ============================================================
# IMPORT LIBRARIES
# ============================================================

# Helps work with files and folders
import os

# Reads YAML configuration files
import yaml


# ============================================================
# GET PROJECT ROOT FOLDER
# ============================================================

# Get location of current Python file

current_file = os.path.abspath(__file__)


# Move upward:

# yolov8
# ↓
# python_codes
# ↓
# IVF_Project

BASE_DIR = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        current_file
                    )
                )
            )


# ============================================================
# DATASET LOCATION
# ============================================================

dataset_path = os.path.join(

    BASE_DIR,

    "datasets",

    "yolo_dataset"

)


# ============================================================
# CHECK IF DATASET EXISTS
# ============================================================

print(

    "Dataset Exists:",

    os.path.exists(dataset_path)

)


# ============================================================
# GET TRAIN IMAGE LIST
# ============================================================

train_images = os.listdir(

    os.path.join(

        dataset_path,

        "images",

        "train"

    )

)


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print()

print("Total train images:")

print(

    len(train_images)

)

print()

print("Sample images:")

print(

    train_images[:5]

)


# ============================================================
# READ DATASET YAML
# ============================================================

yaml_path = os.path.join(

    BASE_DIR,

    "dataset.yaml"

)

with open(

    yaml_path,

    "r"

) as file:

    data_config = yaml.safe_load(file)


print()

print("Dataset configuration:")

print(data_config)