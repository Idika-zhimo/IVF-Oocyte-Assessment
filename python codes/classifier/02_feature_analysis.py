# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd



# ============================================================
# LOAD FEATURE FILE
# ============================================================

df = pd.read_csv(

r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\features.csv"

)



# ============================================================
# SHOW BASIC INFORMATION
# ============================================================

print("\nFirst rows:\n")

print(df.head())



print("\nDataset shape:\n")

print(df.shape)



print("\nMissing values:\n")

print(df.isnull().sum())



print("\nStatistics:\n")

print(df.describe())