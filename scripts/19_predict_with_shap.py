"""
ReferAI - Production-Ready Explainable AI (XAI) Prediction Script
------------------------------------------------------------------
Takes structured clinical inputs, predicts the medical specialist, and prints 
a clear, human-readable breakdown explaining why the model made its decision.
"""
import pandas as pd
import joblib
import json
import os
import shap

MODELS_DIR = "../models"

def run_production_prediction():
    print("Loading model and configuration...")
    model = joblib.load(os.path.join(MODELS_DIR, "specialist_model.pkl"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)
    with open(os.path.join(MODELS_DIR, "default_patient_template.json")) as f:
        default_patient = json.load(f)

    # Restored original clean top 10 symptoms list (Urinary removed)
    top_10_symptoms = [
        "Chest_Pain", "Breathlessness", "Palpitations", "Headache", 
        "Dizziness", "Fever", "Cough", "Abdominal_Pain", "Fatigue", "Skin_Rash"
    ]

    print("=" * 65)
    print("ReferAI - Clinical Specialist Recommendation & AI Explanation")
    print("=" * 65)
    print("\nPlease provide the patient details below:\n")

    user_inputs = {}

    # 1. Numeric Vitals & Demographics
    numeric_prompts = {
        "Age": "Enter patient Age (numeric, e.g., 60): ",
        "Height": "Enter patient Height in cm (numeric, e.g., 160): ",
        "Weight": "Enter patient Weight in kg (numeric, e.g., 70): ",
        "Temperature": "Enter patient Temperature (numeric, e.g., 98.6): ",
        "Oxygen_Saturation": "Enter Oxygen Saturation % (numeric, e.g., 98): "
    }

    for col, prompt in numeric_prompts.items():
        matched = [c for c in feature_columns if col.lower() in c.lower()]
        if matched:
            while True:
                val = input(f" {prompt}").strip()
                try:
                    num_val = float(val) if val else default_patient.get(matched[0], 0)
                    for c in matched:
                        user_inputs[c] = num_val
                    break
                except ValueError:
                    print("   ❌ Invalid input! Please enter a valid number.")

    # 2. Gender Selection
    gender_cols = [c for c in feature_columns if "gender" in c.lower() or "sex" in c.lower()]
    if gender_cols:
        while True:
            g_val = input(" Enter Gender (Male / Female): ").strip().capitalize()
            if g_val in ["Male", "Female"]:
                for col in gender_cols:
                    col_lower = col.lower()
                    if "male" in col_lower and "female" not in col_lower:
                        user_inputs[col] = 1 if g_val == "Male" else 0
                    elif "female" in col_lower:
                        user_inputs[col] = 1 if g_val == "Female" else 0
                    else:
                        user_inputs[col] = g_val
                break
            print("   ❌ Please type 'Male' or 'Female'.")

    # 3. Alcohol Status Selection
    alcohol_cols = [c for c in feature_columns if "alcohol" in c.lower() or "drink" in c.lower()]
    if alcohol_cols:
        while True:
            alc_val = input(" Enter Alcohol status (Never / Regular / Occasional): ").strip().capitalize()
            if alc_val in ["Never", "Regular", "Occasional"]:
                for col in alcohol_cols:
                    col_lower = col.lower()
                    if "never" in col_lower:
                        user_inputs[col] = 1 if alc_val == "Never" else 0
                    elif "regular" in col_lower:
                        user_inputs[col] = 1 if alc_val == "Regular" else 0
                    elif "occasional" in col_lower:
                        user_inputs[col] = 1 if alc_val == "Occasional" else 0
                    else:
                        user_inputs[col] = alc_val
                break
            print("   ❌ Please type 'Never', 'Regular', or 'Occasional'.")

    # 4. Prompt for Top 10 Symptoms (Yes / No)
    print("\n--- Top 10 Clinical Symptoms (Answer y/n) ---")
    for symptom in top_10_symptoms:
        actual_col = next((c for c in feature_columns if c.lower() == symptom.lower()), None)
        if actual_col:
            formatted_name = actual_col.replace('_', ' ')
            while True:
                answer = input(f" {formatted_name}? (y/n): ").strip().lower()
                if answer in ["y", "n", ""]:
                    user_inputs[actual_col] = 1 if answer == "y" else 0
                    break
                print("   ❌ Please type 'y' or 'n'.")

    # Build final patient dataframe
    patient = default_patient.copy()
    patient.update(user_inputs)
    patient_df = pd.DataFrame([patient])[feature_columns]

    # Predict Specialist Probabilities
    probs = model.predict_proba(patient_df)[0]
    
    classes = list(label_encoder.classes_)
    general_care_indices = [i for i, c in enumerate(classes) if "general" in c.lower() or "care" in c.lower() or "gp" in c.lower()]
    
    temp_val = user_inputs.get("Temperature", 98.6)
    oxy_val = user_inputs.get("Oxygen_Saturation", 98)
    has_severe = user_inputs.get("Chest_Pain", 0) == 1 or user_inputs.get("Breathlessness", 0) == 1
    
    if general_care_indices and (97.0 <= temp_val <= 99.0) and (oxy_val >= 95) and not has_severe:
        for idx in general_care_indices:
            probs[idx] *= 1.8

    probs = probs / probs.sum()

    result_df = pd.DataFrame({
        "Specialist": classes,
        "Probability_%": (probs * 100).round(1)
    }).sort_values("Probability_%", ascending=False).reset_index(drop=True)

    top_specialist = result_df.iloc[0]["Specialist"]
    top_prob = result_df.iloc[0]["Probability_%"]

    print("\n" + "=" * 65)
    print("===== SPECIALIST RECOMMENDATION =====")
    print("=" * 65)
    print(f" 🎯 Recommended Specialist: **{top_specialist}** ({top_prob}% confidence)")
    
    print("\nAlternative Recommendations:")
    for _, row in result_df.iloc[1:3].iterrows():
        print(f"   • {row['Specialist']} : {row['Probability_%']}%")

    # Calculate SHAP values for explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(patient_df)
    top_specialist_idx = classes.index(top_specialist)
    
    sample_shap = shap_values.values[0, :, top_specialist_idx]
    
    explanation_df = pd.DataFrame({
        "Feature": feature_columns,
        "Value_Entered": patient_df.iloc[0].values,
        "SHAP_Impact": sample_shap
    }).sort_values(by="SHAP_Impact", ascending=False)

    print("\n" + "=" * 65)
    print(f"===== CLINICAL EXPLAINABLE AI (XAI) BREAKDOWN =====")
    print("=" * 65)
    print(f"Why the model recommended '{top_specialist}' in plain English:\n")
    
    driving_factors = explanation_df[(explanation_df["SHAP_Impact"] > 0) & (explanation_df["Value_Entered"] > 0)].head(5)
    
    for _, row in driving_factors.iterrows():
        feat = row["Feature"].replace("_", " ")
        val = row["Value_Entered"]
        impact = row["SHAP_Impact"]
        
        if val == 1.0:
            print(f" ✔️ Patient reported **{feat}**")
        else:
            print(f" ✔️ Patient recorded **{feat}** (Value: {val})")
            
        print(f"    └─ This key finding strongly supports routing the patient to a **{top_specialist}** (Impact Score: +{impact:.4f}).\n")

    print("✅ Analysis completed successfully!")

if __name__ == "__main__":
    run_production_prediction()