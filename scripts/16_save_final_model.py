import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---- Load FULL training data (we retrain fresh here to keep this script self-contained) ----
X_train = pd.read_csv("../data/X_train.csv")
y_train = pd.read_csv("../data/y_train.csv").squeeze()

print(f"Training final model on {X_train.shape[0]} patients, {X_train.shape[1]} features...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

print("✅ Training complete")

# ---- Save as the official filename ----
joblib.dump(model, "../models/specialist_model.pkl")
print("✅ Model saved as ../models/specialist_model.pkl")
