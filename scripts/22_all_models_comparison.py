"""
ReferAI - All Models Comparison (Random Forest, Logistic Regression, Decision Tree, Gradient Boosting)
--------------------------------------------------------------------------------------------------
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

DATA_DIR = "../data"

def run_all_models():
    print("Loading dataset...")
    csv_path = None
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                break
    
    if not csv_path:
        print("Error: Dataset CSV file not found.")
        return

    df = pd.read_csv(csv_path)
    target_col = "Specialist" if "Specialist" in df.columns else df.columns[-1]
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dictionary of all 4 models with fast configuration parameters
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=30, random_state=42)
    }

    print("\nTraining and evaluating all models...")
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100
        results.append({"Model": name, "Accuracy_%": round(acc, 2)})
        print(f" ✔️ Finished: {name} ({acc:.2f}%)")

    results_df = pd.DataFrame(results).sort_values(by="Accuracy_%", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 50)
    print("===== FINAL ALL-MODEL COMPARISON SUMMARY =====")
    print("=" * 50)
    print(results_df.to_string(index=False))
    print("=" * 50)

if __name__ == "__main__":
    run_all_models()