# ============================================================
# IMPORT LIBRARIES
# ============================================================

# pandas is used for reading and editing CSV files

import pandas as pd



# ============================================================
# LOAD FEATURE FILE
# ============================================================

# Read previously created features.csv

df = pd.read_csv(

r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\features.csv"

)



# ============================================================
# FIND MEDIAN VALUES
# ============================================================

# Median = middle value

# We use median because:
# it is less affected by extreme values

median_circularity = df[

    "circularity"

].median()



median_total_area = df[

    "total_area"

].median()



# ============================================================
# CREATE LABELS
# ============================================================

# Rule:

# If both conditions are true:
# label = 1

# Otherwise:
# label = 0

df["label"] = (

    (df["circularity"] > median_circularity)

    &

    (df["total_area"] > median_total_area)

).astype(int)



# ============================================================
# SAVE NEW CSV
# ============================================================

save_path = (

r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\features_with_labels.csv"

)



df.to_csv(

    save_path,

    index=False
)



# ============================================================
# PRINT RESULTS
# ============================================================

print("="*50)

print("Labels created successfully")

print("="*50)



print("\nLabel counts:\n")

print(

    df["label"].value_counts()

)



print("\nSaved at:\n")

print(save_path)



print("\nFirst rows:\n")

print(df.head())