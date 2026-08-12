import pandas as pd
import joblib

model = joblib.load("../models/specialist_model_temp.pkl")
label_encoder = joblib.load("../models/label_encoder.pkl")

X_test = pd.read_csv("../data/X_test.csv")
y_test = pd.read_csv("../data/y_test.csv").squeeze()

probabilities = model.predict_proba(X_test)
class_names = label_encoder.classes_


def show_top3(patient_index):
    patient_probs = probabilities[patient_index]

    prob_df = pd.DataFrame({
        "Specialist": class_names,
        "Probability_%": (patient_probs * 100).round(1)
    }).sort_values("Probability_%", ascending=False).reset_index(drop=True)

    print("===== Full Probability Breakdown =====")
    for _, row in prob_df.iterrows():
        print(f"{row['Specialist']:<20} {row['Probability_%']}%")

    top_specialist = prob_df.iloc[0]["Specialist"]
    print(f"\nTop Recommendation:\n{top_specialist}")

    print("\nTop 3:")
    for i in range(3):
        specialist = prob_df.iloc[i]["Specialist"]
        prob = prob_df.iloc[i]["Probability_%"]
        print(f"{i+1}. {specialist} – {prob}%")

    actual = label_encoder.inverse_transform([y_test.iloc[patient_index]])[0]
    print(f"\n(Actual specialist in data: {actual})")


# ---- Show for a few different sample patients ----
for idx in [0, 1, 5]:
    print(f"\n{'='*50}")
    print(f"PATIENT #{idx}")
    print('='*50)
    show_top3(idx)