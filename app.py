# ============================================================
# IMPORT LIBRARIES
# ============================================================

from flask import (

    Flask,
    render_template,
    request

)

from werkzeug.utils import secure_filename

from ultralytics import YOLO

import os
import shutil
import glob
import uuid
import numpy as np



# ============================================================
# GET CURRENT PROJECT DIRECTORY
# ============================================================

BASE_DIR=os.path.dirname(

    os.path.abspath(__file__)

)



# ============================================================
# CREATE FLASK APP
# ============================================================

app=Flask(__name__)



# ============================================================
# CREATE FOLDERS
# ============================================================

UPLOAD_FOLDER=os.path.join(

    BASE_DIR,
    "static",
    "uploads"

)


PREDICTION_FOLDER=os.path.join(

    BASE_DIR,
    "static",
    "predictions"

)



os.makedirs(

UPLOAD_FOLDER,

exist_ok=True

)



os.makedirs(

PREDICTION_FOLDER,

exist_ok=True

)



# ============================================================
# LOAD YOLO MODEL
# ============================================================

MODEL_PATH=os.path.join(

BASE_DIR,

"models",

"yolov8",

"best.pt"

)



model=YOLO(

MODEL_PATH

)



# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")

def home():

    return render_template(

        "index.html"

    )



# ============================================================
# PREDICTION
# ============================================================

@app.route(

"/predict",

methods=["POST"]

)

def predict():



    # ========================================================
    # GET IMAGE
    # ========================================================

    file=request.files["image"]



    # ========================================================
    # SAFE FILE NAME
    # ========================================================

    filename=secure_filename(

        file.filename

    )



    # ========================================================
    # UNIQUE ID
    # ========================================================

    unique_id=str(

        uuid.uuid4()

    )[:8]



    # ========================================================
    # CREATE UNIQUE NAME
    # ========================================================

    unique_upload_name=(

        unique_id

        +

        "_"

        +

        filename

    )



    # ========================================================
    # SAVE UPLOADED IMAGE
    # ========================================================

    filepath=os.path.join(

        UPLOAD_FOLDER,

        unique_upload_name

    )



    file.save(

        filepath

    )



    # ========================================================
    # RUN YOLO
    # ========================================================

    results=model.predict(

        source=filepath,

        save=True,

        conf=0.10,

        exist_ok=True

    )



    # ========================================================
    # GET SAVE DIRECTORY
    # ========================================================

    save_dir=str(

        results[0].save_dir

    )



    # ========================================================
    # FIND SAVED IMAGE
    # ========================================================

    image_files=glob.glob(

        os.path.join(

            save_dir,

            "*.jpg"

        )

    )


    image_files+=glob.glob(

        os.path.join(

            save_dir,

            "*.png"

        )

    )



    latest_prediction=max(

        image_files,

        key=os.path.getctime

    )



    # ========================================================
    # SAVE PREDICTION WITH UNIQUE NAME
    # ========================================================

    prediction_filename=(

        "pred_"

        +

        unique_id

        +

        "_"

        +

        filename

    )



    final_prediction=os.path.join(

        PREDICTION_FOLDER,

        prediction_filename

    )



    shutil.copy(

        latest_prediction,

        final_prediction

    )



    # ========================================================
    # EXTRACT DETECTED CLASSES
    # ========================================================

    names=model.names



    detected=[]



    for box in results[0].boxes:

        class_id=int(

            box.cls[0]

        )



        detected.append(

            names[class_id]

        )



    # ========================================================
    # QUALITY LOGIC
    # ========================================================

    polar_present=0

    zona_present=0

    cytoplasm_present=0



    if "polar_body" in detected:

        polar_present=1



    if "zona_pellucida" in detected:

        zona_present=1



    if "cytoplasm" in detected:

        cytoplasm_present=1



    # ========================================================
    # SCORE SYSTEM
    # ========================================================

    score=0



    if polar_present:

        score+=40



    if zona_present:

        score+=30



    if cytoplasm_present:

        score+=30



    confidence=score



    # ========================================================
    # FINAL QUALITY
    # ========================================================

    if score>=80:

        quality="Good Quality"



        reason="""

✓ Polar body detected

✓ Cytoplasm detected

✓ Zona detected

✓ Healthy morphology

"""



    elif score>=40:

        quality="Poor Quality"



        reason="""

⚠ Missing structures

⚠ Weak morphology indicators

"""



    else:

        quality="Low Image Quality"



        reason="""

✗ Critical structures missing

✗ Image quality insufficient

"""



    uploaded_path=(

        "/static/uploads/"

        +

        unique_upload_name

    )
    
    prediction_path=(

        "/static/predictions/"

        +

        prediction_filename

    )



    return render_template(

        "index.html",

        uploaded=uploaded_path,

        prediction=prediction_path,

        quality=quality,

        confidence=confidence,

        reason=reason

    )



# ============================================================
# RUN FLASK
# ============================================================

if __name__=="__main__":

    app.run(

        debug=True

    )