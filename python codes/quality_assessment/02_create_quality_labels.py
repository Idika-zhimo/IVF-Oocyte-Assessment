# ============================================================
# IMPORT LIBRARY
# ============================================================

# pandas
# Used for reading and editing CSV files

import pandas as pd


# ============================================================
# LOAD FEATURE DATA
# ============================================================

# Read quality features created earlier

df = pd.read_csv(

r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\quality_features.csv"

)


# ============================================================
# CREATE EMPTY LABEL LIST
# ============================================================

# Labels:

# 0 = Poor Quality
# 1 = Good Quality
# 2 = Low Image Quality / Uncertain

labels=[]



# ============================================================
# LOOP THROUGH EACH IMAGE
# ============================================================

for _,row in df.iterrows():

    # ========================================================
    # EXTRACT FEATURES
    # ========================================================

    total_area=row["total_area"]

    polar_present=row["polar_present"]

    circularity=row["circularity"]

    zona_ratio=row["zona_ratio"]

    polar_ratio=row["polar_ratio"]



    # ========================================================
    # LOW IMAGE QUALITY / UNCERTAIN
    # ========================================================

    # Cases:

    # Extremely small object
    # Extremely large object
    # Strange shape

    if (

        total_area<30000

        or

        total_area>250000

        or

        circularity<0.50

    ):

        label=2



    # ========================================================
    # POOR QUALITY
    # ========================================================

    # Strong negative indicators

    elif (

        polar_present==0

        or

        zona_ratio<0.30

        or

        polar_ratio<0.005

    ):

        label=0



    # ========================================================
    # GOOD QUALITY
    # ========================================================

    else:

        label=1



    labels.append(label)



# ============================================================
# ADD LABEL COLUMN
# ============================================================

df["label"]=labels



# ============================================================
# SAVE CSV
# ============================================================

save_path=(

r"C:\Users\idika\Documents\TRYONE\IVF_Project-latest\IVF_Project\quality_features_labels.csv"

)


df.to_csv(

save_path,

index=False
)



# ============================================================
# PRINT RESULTS
# ============================================================

print("="*50)

print("Quality labels created")

print("="*50)

print()

print("Meaning:")

print("0 = Poor Quality")

print("1 = Good Quality")

print("2 = Low Image Quality")

print()

print(df["label"].value_counts())

print()

print("Saved at:")

print(save_path)