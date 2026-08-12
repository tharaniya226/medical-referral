import pandas as pd
import joblib
import json

print("===== Loading everything fresh, as if starting a new session =====\n")

# ---- Load the saved model ----
model = joblib.load("../models/specialist_model.pkl")
print("✅ Loaded specialist_model.pkl")

# ---- Load the label encoder ----
label_encoder = joblib.load("../models/label_encoder.pkl")
print("✅ Loaded label_encoder.pkl")

# ---- Load feature column order ----
with open("../models/feature_columns.json") as f:
    feature_columns = json.load(f)
print(f"✅ Loaded feature_columns.json ({len(feature_columns)} columns)")

# ---- Load default patient template ----
with open("../models/default_patient_template.json") as f:
    default_patient = json.load(f)
print("✅ Loaded default_patient_template.json")

# ---- Load top 10 symptoms ----
with open("../models/top_10_symptoms.json") as f:
    top_10_symptoms = json.load(f)
print(f"✅ Loaded top_10_symptoms.json: {top_10_symptoms}")

# ---- Now run a real test prediction using ONLY these freshly loaded files ----
print("\n===== Test Prediction: patient with Chest_Pain + Breathlessness =====")

test_patient = default_patient.copy()
test_patient["Chest_Pain"] = 1
test_patient["Breathlessness"] = 1

patient_df = pd.DataFrame([test_patient])[feature_columns]

probs = model.predict_proba(patient_df)[0]
class_names = label_encoder.classes_

prob_df = pd.DataFrame({
    "Specialist": class_names,
    "Probability_%": (probs * 100).round(1)
}).sort_values("Probability_%", ascending=False).reset_index(drop=True)

print("\nTop 3:")
for i in range(3):
    print(f"{i+1}. {prob_df.iloc[i]['Specialist']} – {prob_df.iloc[i]['Probability_%']}%")

print("\n✅ SUCCESS: Model + all preprocessing objects load and work correctly from saved files.")