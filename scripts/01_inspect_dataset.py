import pandas as pd
import os
import glob

# Automatically find the Excel file inside the data folder,
# no matter its exact name
data_folder = "../data"
excel_files = glob.glob(os.path.join(data_folder, "*.xlsx"))

print("Excel files found in data folder:", excel_files)

if len(excel_files) == 0:
    print("ERROR: No .xlsx file found in the data folder!")
else:
    DATA_PATH = excel_files[0]
    print(f"Using file: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH, sheet_name="Dataset")

    print("\n===== SHAPE =====")
    print(df.shape)

    print("\n===== COLUMN NAMES =====")
    print(list(df.columns))

    print("\n===== MISSING VALUES PER COLUMN =====")
    print(df.isnull().sum().sum(), "total missing values")

    print("\n===== TARGET COLUMN DISTRIBUTION (Specialist) =====")
    print(df["Specialist"].value_counts())