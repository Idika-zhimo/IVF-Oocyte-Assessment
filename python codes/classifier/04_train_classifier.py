# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

# pandas:
# used to read CSV files

import pandas as pd


# train_test_split:
# used to split data into train and test sets

from sklearn.model_selection import train_test_split


# RandomForestClassifier:
# classification algorithm

from sklearn.ensemble import RandomForestClassifier


# Metrics used to evaluate performance

from sklearn.metrics import (

    accuracy_score,
    roc_auc_score,
    classification_report
)



# ============================================================
# LOAD DATASET
# ============================================================

# Read CSV containing features + labels

df = pd.read_csv(

r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\features_with_labels.csv"

)



# ============================================================
# SELECT INPUT FEATURES (X)
# ============================================================

# Remove image name and label

# Remaining columns become model inputs

X = df.drop(

    columns=[

        "image_name",
        "label"

    ]
)



# ============================================================
# SELECT TARGET VARIABLE (y)
# ============================================================

# Model tries to predict this

y = df["label"]



# ============================================================
# SPLIT DATASET
# ============================================================

# 80% → training
# 20% → testing

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)



# ============================================================
# CREATE MODEL
# ============================================================

# n_estimators=100

# Creates 100 decision trees

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42
)



# ============================================================
# TRAIN MODEL
# ============================================================

# Learn patterns from training data

model.fit(

    X_train,
    y_train
)



# ============================================================
# MAKE PREDICTIONS
# ============================================================

predictions = model.predict(

    X_test
)



# ============================================================
# GET PROBABILITY SCORES
# ============================================================

# Needed for AUC

probabilities = model.predict_proba(

    X_test

)[:,1]



# ============================================================
# CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(

    y_test,
    predictions
)



# ============================================================
# CALCULATE AUC
# ============================================================

auc = roc_auc_score(

    y_test,
    probabilities
)



# ============================================================
# PRINT RESULTS
# ============================================================

print("="*50)

print("Random Forest Results")

print("="*50)



print("\nAccuracy:")

print(

    round(
        accuracy,
        4
    )
)



print("\nAUC:")

print(

    round(
        auc,
        4
    )
)



print("\nDetailed Report:\n")

print(

classification_report(

    y_test,
    predictions
)

)
# ============================================================
# IMPORT JOBLIB
# ============================================================

# Used to save trained ML models

import joblib



# ============================================================
# SAVE TRAINED MODEL
# ============================================================

# Path where classifier will be saved

save_path = r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\random_forest_model.pkl"


ifier saved successfully!")

print("\nSaved at:")

print(save_path)
# Save trained model

joblib.dump(

    model,

    save_path
)



# ============================================================
# PRINT SUCCESS MESSAGE
# ============================================================

print("\nClass