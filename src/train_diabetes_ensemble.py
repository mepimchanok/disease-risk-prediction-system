import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier


def eval_model(name, model, X_test, y_test, threshold=0.5):
    # prediction
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        auc = roc_auc_score(y_test, proba)
    else:
        # fallback (กรณีโมเดลไม่มี predict_proba)
        y_pred = model.predict(X_test)
        proba = None
        auc = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n===== {name} (threshold={threshold}) =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    if auc is not None:
        print(f"ROC-AUC  : {auc:.4f}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))


def main():
    # 1) Load prepared dataset
    df = pd.read_csv("data/diabetes_prepared.csv")

    target_col = "Outcome"
    if target_col not in df.columns:
        raise ValueError(f"ไม่เจอคอลัมน์ target '{target_col}' ใน diabetes_prepared.csv")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # save feature names (ไว้ทำเว็บ)
    os.makedirs("models", exist_ok=True)
    with open("models/diabetes_features.json", "w") as f:
        json.dump(X.columns.tolist(), f)

    # 2) Split (สำคัญ: stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)
    print("Positive rate (y=1) train:", float(y_train.mean()))
    print("Positive rate (y=1) test :", float(y_test.mean()))

    # 3) Models (3 ประเภท)
    # (A) Logistic Regression (ต้อง scale)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # (B) Random Forest (bagging)
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    # (C) Gradient Boosting (boosting)
    gb = GradientBoostingClassifier(
        random_state=42
    )

    # 4) Ensemble (Soft Voting)
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
        voting="soft",
        weights=[1, 2, 2]  # ให้ tree models สำคัญขึ้นนิดนึง
    )

    # 5) Train
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    ensemble.fit(X_train, y_train)

    # 6) Evaluate (default threshold = 0.5)
    eval_model("Logistic Regression", lr, X_test, y_test, threshold=0.5)
    eval_model("Random Forest", rf, X_test, y_test, threshold=0.5)
    eval_model("Gradient Boosting", gb, X_test, y_test, threshold=0.5)

    # Ensemble: ลองหลาย threshold (เพราะงานโรคมักอยากได้ recall)
    for th in [0.50, 0.45, 0.40, 0.35]:
        eval_model("Ensemble (Soft Voting)", ensemble, X_test, y_test, threshold=th)

    # 7) Save models
    joblib.dump(lr, "models/diabetes_lr.joblib")
    joblib.dump(rf, "models/diabetes_rf.joblib")
    joblib.dump(gb, "models/diabetes_gb.joblib")
    joblib.dump(ensemble, "models/diabetes_ensemble_voting.joblib")

    print("\n✅ Saved models to /models")
    print("- models/diabetes_lr.joblib")
    print("- models/diabetes_rf.joblib")
    print("- models/diabetes_gb.joblib")
    print("- models/diabetes_ensemble_voting.joblib")
    print("- models/diabetes_features.json")


if __name__ == "__main__":
    main()