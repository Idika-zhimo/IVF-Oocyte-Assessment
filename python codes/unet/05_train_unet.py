# ============================================================
# IMPORT LIBRARIES
# ============================================================

import time

import torch

import torch.nn as nn

import torch.optim as optim

from create_unet import model
from dataset_loader import train_loader


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion=nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer=optim.Adam(

    model.parameters(),

    lr=0.001
)


# ============================================================
# TOTAL EPOCHS
# ============================================================

epochs=10


# ============================================================
# TRAIN LOOP
# ============================================================

for epoch in range(epochs):


    total_loss=0


    start_time=time.time()


    for batch_idx,(images,masks) in enumerate(train_loader):


        optimizer.zero_grad()


        outputs=model(images.float())


        loss=criterion(

            outputs,

            masks.long()
        )


        # Backpropagation

        loss.backward()


        # Update weights

        optimizer.step()


        total_loss+=loss.item()


        progress=((batch_idx+1)/len(train_loader))*100


        if batch_idx%20==0:

            print(

            f"Epoch[{epoch+1}/{epochs}] "

            f"Progress:{progress:.2f}% "

            f"Loss:{loss.item():.4f}"

            )


    avg_loss=total_loss/len(train_loader)

    epoch_time=time.time()-start_time


    print("\n"+"-"*50)

    print(

        "Epoch:",

        epoch+1
    )

    print(

        "Average Loss:",

        avg_loss
    )

    print(

        "Time:",

        epoch_time
    )

    print("-"*50)


print(

    "Training completed"
)