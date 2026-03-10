import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 1) load
df = pd.read_csv("data/heart_disease_uci.csv")

# 2) normalize missing tokens
df = df.replace(["?", "NA", "N/A", "null", "None", ""], np.nan)

# 3) target -> binary (0 = no disease, 1 = disease)
df["num"] = pd.to_numeric(df["num"], errors="coerce")
df["num"] = df["num"].apply(lambda x: 0 if x == 0 else 1)

# 4) drop columns (safe)
df = df.drop(columns=["id", "dataset"], errors="ignore")

# 5) TRUE/FALSE -> 1/0 (robust)
for col in ["fbs", "exang"]:
    if col in df.columns:
        df[col] = (
            df[col].astype(str).str.strip().str.upper()
            .map({"TRUE": 1, "FALSE": 0})
        )

# 6) sex -> 1/0
if "sex" in df.columns:
    df["sex"] = df["sex"].astype(str).str.strip().map({"Male": 1, "Female": 0})

# 7) numeric cols -> numeric
numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 8) categorical -> codes (keep NaN as NaN, ไม่ให้กลายเป็น -1)
cat_cols = ["cp", "restecg", "slope", "thal"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")
        df[col] = df[col].cat.codes.replace(-1, np.nan)

# 9) split X/y
X = df.drop(columns=["num"])
y = df["num"]

# 10) impute (median)
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

# 11) train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 12) scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Positive rate (y=1):", y.mean())
# save prepared data (after impute, before split/scale ก็ได้ แต่แบบนี้คือรวมทั้งหมดแล้ว)
X_all_scaled = scaler.fit_transform(imputer.fit_transform(df.drop(columns=["num"])))
prepared = pd.DataFrame(X_all_scaled, columns=df.drop(columns=["num"]).columns)
prepared["num"] = df["num"].values
prepared.to_csv("data/heart_prepared.csv", index=False)
print("Saved: data/heart_prepared.csv")