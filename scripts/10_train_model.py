import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import time

# ---- Load training data ----
X_train = pd.read_csv("../data/X_train.csv")
y_train = pd.read_csv("../data/y_train.csv").squeeze()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# ---- Define the model ----
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# ---- Train it ----
print("\nTraining model... please wait")
start_time = time.time()

model.fit(X_train, y_train)

end_time = time.time()
print(f"✅ Training complete in {end_time - start_time:.2f} seconds")

# ---- Save the trained model temporarily (we'll do this properly in Step 16) ----
joblib.dump(model, "../models/specialist_model_temp.pkl")
print("\n✅ Model saved temporarily to ../models/specialist_model_temp.pkl")

# ---- Quick sanity check: feature importance ----
importances = pd.Series(model.feature_importances_, index=X_train.columns)
top_10 = importances.sort_values(ascending=False).head(10)

print("\n===== Top 10 most important features =====")
print(top_10)