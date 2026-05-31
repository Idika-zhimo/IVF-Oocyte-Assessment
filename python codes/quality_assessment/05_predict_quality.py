# ============================================================
# IMPORT LIBRARIES
# ============================================================

# joblib
# Used for loading saved model

import joblib


# pandas
# Used for table operations

import pandas as pd



# ============================================================
# LOAD SAVED MODEL
# ============================================================

# Load trained Random Forest model

model=joblib.load(

r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\models\classifier\quality_model.pkl"

)



# ============================================================
# LOAD FEATURES
# ============================================================

# Read extracted features file

df=pd.read_csv(

r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\quality_features.csv"

)



# ============================================================
# PREPARE INPUT FEATURES
# ============================================================

# Remove image filename

# Model only needs numerical values

X=df.drop(

columns=[

'image_name'

]

)



# ============================================================
# MAKE PREDICTIONS
# ============================================================

# Predict class labels

predictions=model.predict(

X

)



# ============================================================
# GET PREDICTION PROBABILITIES
# ============================================================

# Gives confidence values

probabilities=model.predict_proba(

X

)



# ============================================================
# GET MAX CONFIDENCE
# ============================================================

# Highest probability becomes confidence score

confidence=probabilities.max(

axis=1

)



# ============================================================
# ADD RESULTS TO TABLE
# ============================================================

df["predicted_label"]=predictions

df["confidence"]=confidence



# ============================================================
# CONVERT NUMBERS TO WORDS
# ============================================================

# Change:

# 0→Poor
# 1→Good
# 2→Low Image Quality

label_map={

0:"Poor",

1:"Good",

2:"Low Image Quality"

}


df["quality"]=df[

"predicted_label"

].map(

label_map

)



# ============================================================
# SAVE RESULTS
# ============================================================

save_path=(

r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\predicted_quality.csv"

)


df.to_csv(

save_path,

index=False

)



# ============================================================
# SHOW RESULTS
# ============================================================

print()

print("="*50)

print("QUALITY PREDICTION RESULTS")

print("="*50)


print()

print(

df[[

'image_name',

'quality',

'confidence'

]]

.head()

)



print()

print("Saved at:")

print(save_path)