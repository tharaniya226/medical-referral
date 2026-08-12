import pandas as pd
import joblib

# ---- Load model and label encoder ----
model = joblib.load("../models/specialist_model_temp.pkl")
label_encoder = joblib.load("../models/label_encoder.pkl")

# ---- Load test data ----
X_test = pd.read_csv("../data/X_test.csv")
y_test = pd.read_csv("../data/y_test.csv").squeeze()

# ---- Get probability for EVERY class, for EVERY test patient ----
probabilities = model.predict_proba(X_test)

print(f"Probability array shape: {probabilities.shape}")
print("(rows = patients, columns = 9 specialist classes)")

# ---- Show it nicely for the first patient ----
class_names = label_encoder.classes_
first_patient_probs = probabilities[0]

print("\n===== Probability breakdown for Patient #1 (test set) =====")
prob_df = pd.DataFrame({
    "Specialist": class_names,
    "Probability": first_patient_probs
}).sort_values("Probability", ascending=False)

prob_df["Probability_%"] = (prob_df["Probability"] * 100).round(2)
print(prob_df[["Specialist", "Probability_%"]].to_string(index=False))

actual_specialist = label_encoder.inverse_transform([y_test.iloc[0]])[0]
print(f"\nActual specialist for this patient was: {actual_specialist}")