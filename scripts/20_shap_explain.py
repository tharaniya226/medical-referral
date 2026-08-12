"""
ReferAI - SHAP Explainability Script
------------------------------------
Loads the trained Random Forest model and generates SHAP values 
to explain feature importance for patient predictions.
"""
import pandas as pd
import joblib
import json
import os
import shap
import matplotlib.pyplot as plt

MODELS_DIR = "../models"

def run_shap_analysis():
    print("Loading models and features for SHAP analysis...")
    model = joblib.load(os.path.join(MODELS_DIR, "specialist_model.pkl"))
    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)
    with open(os.path.join(MODELS_DIR, "default_patient_template.json")) as f:
        default_patient = json.load(f)

    # Create a sample patient profile (e.g., matching a cardiac/pulmonary presentation)
    sample_patient = default_patient.copy()
    sample_patient["Chest_Pain"] = 1
    sample_patient["Breathlessness"] = 1
    sample_patient["Age"] = 55.0

    sample_df = pd.DataFrame([sample_patient])[feature_columns]

    print("Calculating SHAP values...")
    # Use TreeExplainer for Random Forest models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_df)

    print("\n--- SHAP Analysis Completed Successfully! ---")
    print("SHAP values calculated for the sample patient.")
    print("You can include SHAP summary or waterfall plots in your project presentation slides.")

if __name__ == "__main__":
    run_shap_analysis()