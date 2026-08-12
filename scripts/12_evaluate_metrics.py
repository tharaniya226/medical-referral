import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---- Load everything ----
y_test = pd.read_csv("../data/y_test.csv").squeeze()
y_pred = pd.read_csv("../data/y_pred.csv").squeeze()
label_encoder = joblib.load("../models/label_encoder.pkl")

class_names = label_encoder.classes_

# ---- Accuracy ----
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")

# ---- Precision, Recall, F1 (macro = treats all 9 classes equally) ----
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print(f"Precision (macro avg): {precision:.4f}")
print(f"Recall (macro avg): {recall:.4f}")
print(f"F1-score (macro avg): {f1:.4f}")

# ---- Confusion Matrix ----
print("\n===== Confusion Matrix =====")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print(cm_df)

# ---- Full Classification Report (per-class precision/recall/f1) ----
print("\n===== Classification Report =====")
report = classification_report(y_test, y_pred, target_names=class_names)
print(report)

# ---- Save confusion matrix and report to files for your project documentation ----
cm_df.to_csv("../data/confusion_matrix.csv")
with open("../data/classification_report.txt", "w") as f:
    f.write(report)

print("\n✅ Confusion matrix saved to ../data/confusion_matrix.csv")
print("✅ Classification report saved to ../data/classification_report.txt")
