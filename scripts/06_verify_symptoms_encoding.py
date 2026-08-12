import pandas as pd

df = pd.read_csv("../data/encoded_dataset.csv")

SYMPTOMS = [
    "Chest_Pain", "Breathlessness", "Palpitations", "Headache", "Dizziness",
    "Fever", "Cough", "Sore_Throat", "Abdominal_Pain", "Vomiting", "Diarrhea",
    "Joint_Pain", "Back_Pain", "Skin_Rash", "Vision_Problem", "Ear_Pain",
    "Urinary_Problem", "Fatigue", "Nausea", "Swelling", "Muscle_Pain",
    "Loss_of_Appetite", "Difficulty_Swallowing", "Wheezing", "Nasal_Congestion"
]

MEDICAL_HISTORY = [
    "Diabetes", "Hypertension", "Previous_Heart_Disease", "Previous_Lung_Disease",
    "Previous_Neurological_Disease", "Previous_Gastrointestinal_Disease",
    "Previous_Bone_or_Joint_Disease", "Previous_Skin_Disease",
    "Previous_Eye_Disease", "Previous_ENT_Disease"
]

print("===== Checking Symptom columns are numeric (0/1) =====")
all_good = True
for col in SYMPTOMS:
    unique_vals = set(df[col].unique())
    if unique_vals.issubset({0, 1}):
        pass  # good
    else:
        print(f"⚠️ {col} has unexpected values: {unique_vals}")
        all_good = False

if all_good:
    print(f"✅ All {len(SYMPTOMS)} symptom columns are numeric (0/1).")

print("\n===== Checking Medical History columns are numeric (0/1) =====")
all_good2 = True
for col in MEDICAL_HISTORY:
    unique_vals = set(df[col].unique())
    if unique_vals.issubset({0, 1}):
        pass
    else:
        print(f"⚠️ {col} has unexpected values: {unique_vals}")
        all_good2 = False

if all_good2:
    print(f"✅ All {len(MEDICAL_HISTORY)} medical history columns are numeric (0/1).")

# ---- Show how often each symptom appears, per specialist (sanity check) ----
print("\n===== Sample: how often Chest_Pain=1 appears per specialist =====")
print(df.groupby("Specialist")["Chest_Pain"].mean().sort_values(ascending=False))

print("\n===== Sample: how often Skin_Rash=1 appears per specialist =====")
print(df.groupby("Specialist")["Skin_Rash"].mean().sort_values(ascending=False))
