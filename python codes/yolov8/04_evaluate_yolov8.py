from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__)
                    )
                )
            )

model_path = os.path.join(
    BASE_DIR,
    "models",
    "yolov8",
    "best.pt"
)

yaml_path = os.path.join(
    BASE_DIR,
    "dataset.yaml"
)

model = YOLO(model_path)

results = model.val(data=yaml_path)

print("="*50)
print("YOLOv8 Evaluation Results")
print("="*50)

print("mAP50:", results.box.map50)
print("mAP50-95:", results.box.map)
print("Precision:", results.box.mp)
print("Recall:", results.box.mr)

print("="*50)