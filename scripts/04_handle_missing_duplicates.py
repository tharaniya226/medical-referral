import pandas as pd

df = pd.read_csv("../data/cleaned_dataset.csv")

print(f"Starting shape: {df.shape}")

# ---- Step A: Check missing values per column ----
print("\n===== Missing values per column =====")
missing_counts = df.isnull().sum()
missing_cols = missing_counts[missing_counts > 0]

if len(missing_cols) == 0:
    print("✅ No missing values found in any column.")
else:
    print(f"⚠️ Missing values found in {len(missing_cols)} columns:")
    print(missing_cols)

    # If missing values existed, here's how we WOULD handle them:
    # Numeric columns -> fill with median (robust to outliers)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled missing {col} with median: {median_val}")

    # Categorical/text columns -> fill with mode (most frequent value)
    categorical_cols = df.select_dtypes(include="object").columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"Filled missing {col} with mode: {mode_val}")

# ---- Step B: Check and remove duplicate rows ----
print("\n===== Duplicate rows =====")
dup_count = df.duplicated().sum()
print(f"Duplicate rows found: {dup_count}")

if dup_count > 0:
    df = df.drop_duplicates()
    print(f"✅ Removed {dup_count} duplicate rows.")
else:
    print("✅ No duplicate rows to remove.")

# ---- Step C: Check and remove duplicate Patient_IDs ----
print("\n===== Duplicate Patient_IDs =====")
dup_id_count = df["Patient_ID"].duplicated().sum()
print(f"Duplicate Patient_IDs found: {dup_id_count}")

if dup_id_count > 0:
    df = df.drop_duplicates(subset="Patient_ID", keep="first")
    print(f"✅ Removed {dup_id_count} duplicate Patient_ID rows.")
else:
    print("✅ No duplicate Patient_IDs to remove.")

# ---- Save final version ----
df.to_csv("../data/no_missing_no_duplicates.csv", index=False)
print(f"\n✅ Saved to: ../data/no_missing_no_duplicates.csv")
print(f"Final shape: {df.shape}")