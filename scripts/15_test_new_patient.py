import pandas as pd
import joblib

# ---- Load our trained model + label encoder (our actual pipeline files) ----
model = joblib.load("../models/specialist_model_temp.pkl")
label_encoder = joblib.load("../models/label_encoder.pkl")

X_train = pd.read_csv("../data/X_train.csv")
feature_columns = X_train.columns.tolist()

# ---- Identify symptom-only columns (exclude vitals, age, and one-hot columns) ----
numeric_cols = ["Age", "Height_cm", "Weight_kg", "BMI", "Systolic_BP",
                 "Diastolic_BP", "Heart_Rate", "Temperature", "Oxygen_Saturation"]
onehot_prefixes = ["Gender_", "Smoking_Status_", "Alcohol_Use_"]

symptom_and_history_cols = [
    col for col in feature_columns
    if col not in numeric_cols and not any(col.startswith(p) for p in onehot_prefixes)
]

# ---- Rank by importance, keep top 10 ----
importances = pd.Series(model.feature_importances_, index=feature_columns)
top_10_symptoms = importances[symptom_and_history_cols].sort_values(ascending=False).head(10).index.tolist()

print("===== Top 10 most important symptoms/history (used for this test) =====")
for s in top_10_symptoms:
    print(f"  - {s}")

# ---- DEFAULT VALUES (healthy baseline for everything) ----
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

# ---- Ask the user ONLY about the top 10 symptoms ----
print("\n===== Answer y/n for each (press Enter for 'n' if unsure) =====")
final_patient = default_patient.copy()
for symptom in top_10_symptoms:
    answer = input(f"{symptom.replace('_', ' ')}? (y/n): ").strip().lower()
    final_patient[symptom] = 1 if answer == "y" else 0

# ---- Build the row in the correct column order and predict ----
new_patient_df = pd.DataFrame([final_patient])[feature_columns]

probs = model.predict_proba(new_patient_df)[0]
class_names = label_encoder.classes_

prob_df = pd.DataFrame({
    "Specialist": class_names,
    "Probability_%": (probs * 100).round(1)
}).sort_values("Probability_%", ascending=False).reset_index(drop=True)

print("\n===== Result =====")
print("\nFull Probability Breakdown:")
for _, row in prob_df.iterrows():
    print(f"  {row['Specialist']:<20} {row['Probability_%']}%")

print(f"\nTop Recommendation:\n{prob_df.iloc[0]['Specialist']}")

print("\nTop 3:")
for i in range(3):
    print(f"{i+1}. {prob_df.iloc[i]['Specialist']} – {prob_df.iloc[i]['Probability_%']}%")