import os
import cv2
import numpy as np

folder=r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\datasets\augmented_dataset\masks\train"

all_values=set()

for file in os.listdir(folder):

    path=os.path.join(folder,file)

    mask=cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is not None:

        values=np.unique(mask)

        all_values.update(values)

print("Classes found:")
print(sorted(all_values))