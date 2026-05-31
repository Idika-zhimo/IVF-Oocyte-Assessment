# ============================================================
# IMPORT LIBRARIES
# ============================================================

# pandas
# Used for reading CSV files into table format

import pandas as pd


# joblib
# Used to load previously saved machine learning model

import joblib


# train_test_split
# Used for splitting dataset into training and testing sets

from sklearn.model_selection import train_test_split


# Import evaluation metrics

from sklearn.metrics import (

    accuracy_score,          # Calculates overall accuracy

    classification_report,   # Shows precision, recall and f1-score

    confusion_matrix         # Shows prediction mistakes
)



# ============================================================
# LOAD DATASET
# ============================================================

# Read CSV file containing:

# image_name
# extracted features
# labels

df = pd.read_csv(

r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\quality_features_labels.csv"

)



# ============================================================
# DEFINE INPUT FEATURES (X)
# ============================================================

# Remove columns that should NOT be used

# image_name:
# file name has no biological meaning

# label:
# this is the answer we want to predict

# Remaining columns become input variables

X = df.drop(

columns=[

'image_name',

'label'

]

)



# ============================================================
# DEFINE TARGET VARIABLE (y)
# ============================================================

# Labels:

# 0 → Poor Quality

# 1 → Good Quality

# 2 → Low Image Quality

# This is what model tries to predict

y=df['label']



# ============================================================
# RECREATE SAME TRAIN/TEST SPLIT
# ============================================================

# Why recreate?

# During training:

# model only saw training data

# Evaluation should happen only on unseen test data

# test_size=0.2

# means:

# 20% data for testing

# random_state=42

# keeps same split every run

# stratify=y

# keeps label proportions similar

X_train,X_test,y_train,y_test=train_test_split(

X,

y,

test_size=0.2,

random_state=42,

stratify=y

)



# ============================================================
# LOAD SAVED MODEL
# ============================================================

# Load previously trained Random Forest model

model=joblib.load(

r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\models\classifier\quality_model.pkl"

)



# ============================================================
# MAKE PREDICTIONS
# ============================================================

# Give unseen test data to model

# Model predicts:

# Poor

# Good

# Low Image Quality

predictions=model.predict(

X_test

)



# ============================================================
# CALCULATE ACCURACY
# ============================================================

# Formula:

# Correct predictions
# ---------------------
# Total predictions

accuracy=accuracy_score(

y_test,

predictions

)



# ============================================================
# PRINT EVALUATION RESULTS
# ============================================================

print()

print("="*50)

print("QUALITY MODEL EVALUATION")

print("="*50)



print()

print(

"Accuracy:",

round(

accuracy,

4

)

)



print()

print("Classification Report")

print()



# classification_report gives:

# Precision:
# Out of predicted labels,
# how many were correct

# Recall:
# Out of actual labels,
# how many model found

# F1-score:
# balance between precision and recall

print(

classification_report(

y_test,

predictions

)

)



print()

print("Confusion Matrix")

print()



# Confusion Matrix:

# Rows:

# Actual values

# Columns:

# Predicted values

# Example:

# [[50 2 0]
#  [3 45 1]
#  [0 1 20]]

# Means:

# 50 correct class0

# 2 class0 predicted incorrectly

print(

confusion_matrix(

y_test,

predictions

)

)