import pandas as pd
import os
import glob

data_folder = "../data"
excel_files = glob.glob(os.path.join(data_folder, "*.xlsx"))
DATA_PATH = excel_files[0]

df = pd.read_excel(DATA_PATH, sheet_name="Dataset")

print(f"Starting shape: {df.shape}")

# ---- 1. Strip whitespace from all text columns ----
text_columns = df.select_dtypes(include="object").columns
for col in text_columns:
    df[col] = df[col].astype(str).str.strip()

# ---- 2. Check for unexpected values in Yes/No columns ----
BINARY_YESNO_COLUMNS = [
    "Family_History",
    "Chest_Pain", "Breathlessness", "Palpitations", "Headache", "Dizziness",
    "Fever", "Cough", "Sore_Throat", "Abdominal_Pain", "Vomiting", "Diarrhea",
    "Joint_Pain", "Back_Pain", "Skin_Rash", "Vision_Problem", "Ear_Pain",
    "Urinary_Problem", "Fatigue", "Nausea", "Swelling", "Muscle_Pain",
    "Loss_of_Appetite", "Difficulty_Swallowing", "Wheezing", "Nasal_Congestion",
    "Diabetes", "Hypertension", "Previous_Heart_Disease", "Previous_Lung_Disease",
    "Previous_Neurological_Disease", "Previous_Gastrointestinal_Disease",
    "Previous_Bone_or_Joint_Disease", "Previous_Skin_Disease",
    "Previous_Eye_Disease", "Previous_ENT_Disease"
]

print("\n===== Checking Yes/No columns for unexpected values =====")
problems_found = False
for col in BINARY_YESNO_COLUMNS:
    unique_vals = set(df[col].unique())
    if not unique_vals.issubset({"Yes", "No"}):
        print(f"⚠️ {col} has unexpected values: {unique_vals}")
        problems_found = True

if not problems_found:
    print("✅ All Yes/No columns contain only 'Yes' or 'No'.")

# ---- 3. Check numeric columns for impossible values ----
print("\n===== Checking numeric ranges =====")
checks = {
    "Age": (0, 120),
    "Height_cm": (100, 220),
    "Weight_kg": (20, 200),
    "BMI": (10, 60),
    "Systolic_BP": (70, 220),
    "Diastolic_BP": (40, 140),
    "Heart_Rate": (30, 200),
    "Temperature": (34, 42),
    "Oxygen_Saturation": (70, 100),
}

range_problems = False
for col, (low, high) in checks.items():
    out_of_range = df[(df[col] < low) | (df[col] > high)]
    if len(out_of_range) > 0:
        print(f"⚠️ {col} has {len(out_of_range)} values outside expected range [{low}, {high}]")
        range_problems = True

if not range_problems:
    print("✅ All numeric columns within expected realistic ranges.")

# ---- 4. Check for duplicate rows ----
print("\n===== Checking for duplicate rows =====")
duplicate_count = df.duplicated().sum()
print(f"Duplicate rows found: {duplicate_count}")

# ---- 5. Check for duplicate Patient_IDs ----
print("\n===== Checking for duplicate Patient_ID =====")
duplicate_ids = df["Patient_ID"].duplicated().sum()
print(f"Duplicate Patient_IDs found: {duplicate_ids}")

# ---- Save the cleaned version ----
output_path = "../data/cleaned_dataset.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Cleaned dataset saved to: {output_path}")
print(f"Final shape: {df.shape}")