import pandas as pd
import joblib

# ---- Load the trained model ----
model = joblib.load("../models/specialist_model_temp.pkl")

# ---- Load test data ----
X_test = pd.read_csv("../data/X_test.csv")
y_test = pd.read_csv("../data/y_test.csv").squeeze()

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# ---- Make predictions on the test set ----
y_pred = model.predict(X_test)

print(f"\nPredictions made: {len(y_pred)}")

# ---- Quick look: compare first 10 actual vs predicted ----
label_encoder = joblib.load("../models/label_encoder.pkl")

comparison = pd.DataFrame({
    "Actual": label_encoder.inverse_transform(y_test[:10]),
    "Predicted": label_encoder.inverse_transform(y_pred[:10])
})
print("\n===== First 10 predictions vs actual =====")
print(comparison)

# ---- Save predictions for Step 12 (metrics) ----
pd.Series(y_pred).to_csv("../data/y_pred.csv", index=False)
print("\n✅ Predictions saved to ../data/y_pred.csv")