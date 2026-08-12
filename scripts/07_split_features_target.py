import pandas as pd

df = pd.read_csv("../data/encoded_dataset.csv")

print(f"Full dataset shape: {df.shape}")

# ---- Define target ----
y = df["Specialist_Encoded"]

# ---- Define features (drop ID + both versions of target) ----
X = df.drop(columns=["Patient_ID", "Specialist", "Specialist_Encoded"])

print(f"\nX (features) shape: {X.shape}")
print(f"y (target) shape: {y.shape}")

print(f"\nFeature columns ({X.shape[1]} total):")
print(list(X.columns))

print(f"\ny value counts (encoded):")
print(y.value_counts().sort_index())

# ---- Save X and y separately for the next step ----
X.to_csv("../data/X_features.csv", index=False)
y.to_csv("../data/y_target.csv", index=False)

print("\n✅ Saved X to ../data/X_features.csv")
print("✅ Saved y to ../data/y_target.csv")