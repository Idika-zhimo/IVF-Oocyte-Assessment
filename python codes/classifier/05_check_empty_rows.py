# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd



# ============================================================
# LOAD FEATURES FILE
# ============================================================

df = pd.read_csv(

r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\features.csv"

)



# ============================================================
# FIND SUSPICIOUS ROWS
# ============================================================

empty_rows = df[

(df["cytoplasm_area"]==0)

|

(df["zona_area"]==0)

]



# ============================================================
# PRINT RESULTS
# ============================================================

print("\nNumber of suspicious rows:\n")

print(len(empty_rows))



print("\nFirst suspicious rows:\n")

print(empty_rows.head())