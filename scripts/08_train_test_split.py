import pandas as pd
from sklearn.model_selection import train_test_split

X = pd.read_csv("../data/X_features.csv")
y = pd.read_csv("../data/y_target.csv").squeeze()  # squeeze turns it back into a single column (Series)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ---- Split: 80% train, 20% test, stratified by y so class balance is preserved ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,      # fixes the randomness so results are reproducible
    stratify=y             # keeps the same class proportions in both sets
)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

print("\n===== y_train class distribution =====")
print(y_train.value_counts().sort_index())

print("\n===== y_test class distribution =====")
print(y_test.value_counts().sort_index())

# ---- Save all four pieces for the next step ----
X_train.to_csv("../data/X_train.csv", index=False)
X_test.to_csv("../data/X_test.csv", index=False)
y_train.to_csv("../data/y_train.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)

print("\n✅ Saved X_train, X_test, y_train, y_test to ../data/")
