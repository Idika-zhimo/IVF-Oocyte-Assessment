# ============================================================
# IMPORT LIBRARIES
# ============================================================

# pandas
# Used for reading CSV files and handling table data

import pandas as pd


# joblib
# Used to save trained machine learning models

import joblib


# train_test_split
# Used for splitting dataset into training and testing parts

from sklearn.model_selection import train_test_split


# RandomForestClassifier
# Machine learning algorithm that uses multiple decision trees

from sklearn.ensemble import RandomForestClassifier


# Metrics used for evaluating model performance

from sklearn.metrics import (

    accuracy_score,          # Calculates overall prediction accuracy

    classification_report    # Gives precision, recall, f1-score

)



# ============================================================
# LOAD DATASET
# ============================================================

# Read previously created CSV file

# This file contains:

# image_name
# extracted features
# labels

df = pd.read_csv(

r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\quality_features_labels.csv"

)



# ============================================================
# DEFINE INPUT FEATURES (X)
# ============================================================

# Remove columns that should NOT be used for learning

# image_name:
# filename has no biological meaning

# label:
# label is target output

# Remaining columns become input features

X = df.drop(

columns=[

'image_name',

'label'

]

)



# ============================================================
# DEFINE TARGET VARIABLE (y)
# ============================================================

# y is what we want the model to predict

# Labels:

# 0 = Poor Quality
# 1 = Good Quality
# 2 = Low Image Quality

y=df['label']



# ============================================================
# SPLIT TRAINING AND TEST DATA
# ============================================================

# Split dataset into:

# 80% training data
# Used for learning patterns

# 20% testing data
# Used for checking performance

# random_state=42

# Gives same split every run

# stratify=y

# Keeps label proportions same

X_train,X_test,y_train,y_test=train_test_split(

X,

y,

test_size=0.2,

random_state=42,

stratify=y

)



# ============================================================
# CREATE RANDOM FOREST MODEL
# ============================================================

# Random Forest creates multiple decision trees

# n_estimators=200

# Create 200 trees

# class_weight='balanced'

# Important because our labels are imbalanced

# Current data:

# Poor = 3200
# Good = 36
# Low Quality = 24

# Without balancing:

# model may predict everything as Poor

model=RandomForestClassifier(

n_estimators=200,

class_weight='balanced',

random_state=42

)



# ============================================================
# TRAIN MODEL
# ============================================================

# fit()

# Model learns relationships between:

# Features
# ↓
# Labels

print()

print("Training model...")


model.fit(

X_train,

y_train

)



# ============================================================
# MAKE PREDICTIONS
# ============================================================

# Model predicts labels for test data

predictions=model.predict(

X_test

)



# ============================================================
# CALCULATE ACCURACY
# ============================================================

# Formula:

# Correct predictions
# -------------------
# Total predictions

accuracy=accuracy_score(

y_test,

predictions

)



# ============================================================
# PRINT RESULTS
# ============================================================

print()

print("="*50)

print("QUALITY CLASSIFIER RESULTS")

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


print(

classification_report(

y_test,

predictions

)

)



# ============================================================
# SAVE TRAINED MODEL
# ============================================================

# Define where model will be saved

save_path=(

r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\models\classifier\quality_model.pkl"

)



# Save trained model

joblib.dump(

model,

save_path

)



# ============================================================
# PRINT SUCCESS MESSAGE
# ============================================================

print()

print("Model saved successfully")

print()

print(save_path)