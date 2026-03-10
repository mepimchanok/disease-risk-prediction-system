import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# โหลด dataset
df = pd.read_csv("data/diabetes.csv")

# แยก feature/target
y = df["Outcome"]
X = df.drop(columns=["Outcome"])

# ทำให้ข้อมูล "ไม่สมบูรณ์" แบบที่เจอบ่อยในชุดนี้: 0 = missing ในบางคอลัมน์
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_as_missing:
    X[col] = X[col].replace(0, np.nan)

# เติมค่าว่าง
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Positive rate (y=1):", float(y.mean()))

# เซฟ prepared csv (ทั้งชุดหลัง impute+scale)
X_all_scaled = scaler.fit_transform(X_imputed)
prepared = pd.DataFrame(X_all_scaled, columns=X.columns)
prepared["Outcome"] = y.values
prepared.to_csv("data/diabetes_prepared.csv", index=False)
print("Saved: data/diabetes_prepared.csv")