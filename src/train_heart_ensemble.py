import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier


def eval_model(name, model, X_test, y_test, threshold=0.5):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        auc = roc_auc_score(y_test, proba)
    else:
        y_pred = model.predict(X_test)
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
    df = pd.read_csv("data/heart_prepared.csv")

    target_col = "num"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)
    print("Positive rate train:", float(y_train.mean()))
    print("Positive rate test :", float(y_test.mean()))

    lr = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)

    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced"
    )

    gb = GradientBoostingClassifier(random_state=42)

    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
        voting="soft"
    )

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    ensemble.fit(X_train, y_train)

    # แนะนำให้โชว์ทั้ง 0.5 และ threshold ที่ต่ำลง (เน้นจับผู้ป่วยให้เจอ)
    eval_model("Logistic Regression", lr, X_test, y_test, threshold=0.5)
    eval_model("Random Forest", rf, X_test, y_test, threshold=0.5)
    eval_model("Gradient Boosting", gb, X_test, y_test, threshold=0.5)

    for th in [0.50, 0.45, 0.40, 0.35]:
        eval_model("Ensemble (Soft Voting)", ensemble, X_test, y_test, threshold=th)

    os.makedirs("models", exist_ok=True)
    joblib.dump(lr, "models/heart_lr.joblib")
    joblib.dump(rf, "models/heart_rf.joblib")
    joblib.dump(gb, "models/heart_gb.joblib")
    joblib.dump(ensemble, "models/heart_ensemble_voting.joblib")

    meta = {
        "target_col": target_col,
        "feature_cols": list(X.columns),
        "seed": 42
    }
    with open("models/heart_features.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n✅ Saved:")
    print("- models/heart_lr.joblib")
    print("- models/heart_rf.joblib")
    print("- models/heart_gb.joblib")
    print("- models/heart_ensemble_voting.joblib")
    print("- models/heart_features.json")


if __name__ == "__main__":
    main()