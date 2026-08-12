import pandas as pd
import joblib
import json

# ---- 1. Save the exact column order the model expects ----
X_train = pd.read_csv("../data/X_train.csv")
feature_columns = X_train.columns.tolist()

with open("../models/feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)

print(f"✅ Saved {len(feature_columns)} feature column names to ../models/feature_columns.json")

# ---- 2. Save the default/healthy-baseline patient template ----
default_patient = {
    "Age": 40, "Height_cm": 165, "Weight_kg": 65, "BMI": 24.0,
    "Family_History": 0,
    "Chest_Pain": 0, "Breathlessness": 0, "Palpitations": 0, "Headache": 0,
    "Dizziness": 0, "Fever": 0, "Cough": 0, "Sore_Throat": 0,
    "Abdominal_Pain": 0, "Vomiting": 0, "Diarrhea": 0, "Joint_Pain": 0,
    "Back_Pain": 0, "Skin_Rash": 0, "Vision_Problem": 0, "Ear_Pain": 0,
    "Urinary_Problem": 0, "Fatigue": 0, "Nausea": 0, "Swelling": 0,
    "Muscle_Pain": 0, "Loss_of_Appetite": 0, "Difficulty_Swallowing": 0,
    "Wheezing": 0, "Nasal_Congestion": 0,
    "Diabetes": 0, "Hypertension": 0, "Previous_Heart_Disease": 0,
    "Previous_Lung_Disease": 0, "Previous_Neurological_Disease": 0,
    "Previous_Gastrointestinal_Disease": 0, "Previous_Bone_or_Joint_Disease": 0,
    "Previous_Skin_Disease": 0, "Previous_Eye_Disease": 0,
    "Previous_ENT_Disease": 0,
    "Systolic_BP": 118, "Diastolic_BP": 76, "Heart_Rate": 75,
    "Temperature": 36.8, "Oxygen_Saturation": 98,
    "Gender_Female": 0, "Gender_Male": 1,
    "Smoking_Status_Current": 0, "Smoking_Status_Former": 0, "Smoking_Status_Never": 1,
    "Alcohol_Use_Never": 1, "Alcohol_Use_Occasional": 0, "Alcohol_Use_Regular": 0,
}

with open("../models/default_patient_template.json", "w") as f:
    json.dump(default_patient, f, indent=2)

print("✅ Saved default patient template to ../models/default_patient_template.json")

# ---- 3. Confirm label encoder already exists (from Step 5) ----
label_encoder = joblib.load("../models/label_encoder.pkl")
print(f"✅ Label encoder confirmed: {len(label_encoder.classes_)} classes")

# ---- 4. Also save the top-10-important-symptoms list (used in Step 15/19) ----
model = joblib.load("../models/specialist_model.pkl")

numeric_cols = ["Age", "Height_cm", "Weight_kg", "BMI", "Systolic_BP",
                 "Diastolic_BP", "Heart_Rate", "Temperature", "Oxygen_Saturation"]
onehot_prefixes = ["Gender_", "Smoking_Status_", "Alcohol_Use_"]

symptom_and_history_cols = [
    col for col in feature_columns
    if col not in numeric_cols and not any(col.startswith(p) for p in onehot_prefixes)
]

importances = pd.Series(model.feature_importances_, index=feature_columns)
top_10_symptoms = importances[symptom_and_history_cols].sort_values(ascending=False).head(10).index.tolist()

with open("../models/top_10_symptoms.json", "w") as f:
    json.dump(top_10_symptoms, f, indent=2)

print(f"✅ Saved top 10 important symptoms to ../models/top_10_symptoms.json")
print(top_10_symptoms)
