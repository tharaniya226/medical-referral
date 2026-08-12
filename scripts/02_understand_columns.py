import pandas as pd
import os
import glob

data_folder = "../data"
excel_files = glob.glob(os.path.join(data_folder, "*.xlsx"))
DATA_PATH = excel_files[0]

df = pd.read_excel(DATA_PATH, sheet_name="Dataset")

TARGET_COLUMN = "Specialist"
ID_COLUMN = "Patient_ID"

NUMERIC_COLUMNS = [
    "Age", "Height_cm", "Weight_kg", "BMI",
    "Systolic_BP", "Diastolic_BP", "Heart_Rate",
    "Temperature", "Oxygen_Saturation"
]

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

MULTICLASS_CATEGORICAL_COLUMNS = ["Gender", "Smoking_Status", "Alcohol_Use"]

all_listed = (
    [ID_COLUMN] + NUMERIC_COLUMNS + BINARY_YESNO_COLUMNS +
    MULTICLASS_CATEGORICAL_COLUMNS + [TARGET_COLUMN]
)

actual_columns = list(df.columns)

missing_from_list = set(actual_columns) - set(all_listed)
extra_in_list = set(all_listed) - set(actual_columns)

print("Columns in dataset but NOT categorized above:", missing_from_list)
print("Columns categorized above but NOT in dataset:", extra_in_list)

if not missing_from_list and not extra_in_list:
    print("\n✅ All 50 columns accounted for correctly.")
else:
    print("\n⚠️ Mismatch found — check the lists above.")

print(f"\nTotal columns: {len(actual_columns)}")
print(f"Target classes: {df[TARGET_COLUMN].nunique()} -> {sorted(df[TARGET_COLUMN].unique())}")