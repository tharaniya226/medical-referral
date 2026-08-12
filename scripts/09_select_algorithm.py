from sklearn.ensemble import RandomForestClassifier

# n_estimators = number of decision trees in the forest
# random_state = fixes randomness so results are reproducible
# n_jobs=-1 = uses all CPU cores to train faster
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("Selected algorithm: Random Forest Classifier")
print(f"\nModel settings:")
print(model.get_params())