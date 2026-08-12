import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("../data/no_missing_no_duplicates.csv")

print(f"Starting shape: {df.shape}")

# ---- 1. Encode binary Yes/No columns as 1/0 ----
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

for col in BINARY_YESNO_COLUMNS:
    df[col] = df[col].map({"Yes": 1, "No": 0})

print("✅ Binary Yes/No columns encoded as 1/0")

# ---- 2. One-Hot Encode multi-class categorical columns ----
MULTICLASS_CATEGORICAL_COLUMNS = ["Gender", "Smoking_Status", "Alcohol_Use"]

df = pd.get_dummies(df, columns=MULTICLASS_CATEGORICAL_COLUMNS, drop_first=False)

print("✅ Gender, Smoking_Status, Alcohol_Use one-hot encoded")

# ---- 3. Label Encode the target column (Specialist) ----
label_encoder = LabelEncoder()
df["Specialist_Encoded"] = label_encoder.fit_transform(df["Specialist"])

print("✅ Specialist encoded as numbers")
print("\nLabel mapping (number -> specialist name):")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"  {i} -> {class_name}")

# ---- Save the label encoder for later use (Step 17) ----
import joblib
import os
os.makedirs("../models", exist_ok=True)
joblib.dump(label_encoder, "../models/label_encoder.pkl")
print("\n✅ Label encoder saved to ../models/label_encoder.pkl")

# ---- Save the fully encoded dataset ----
df.to_csv("../data/encoded_dataset.csv", index=False)
print(f"\n✅ Encoded dataset saved to: ../data/encoded_dataset.csv")
print(f"Final shape: {df.shape}")
print(f"\nColumn list after encoding:")
print(list(df.columns))