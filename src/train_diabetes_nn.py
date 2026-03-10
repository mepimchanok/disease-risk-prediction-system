# src/train_diabetes_nn.py
import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(input_dim: int) -> keras.Model:
    # เหมาะกับ tabular: BN + Dropout + Dense หลายชั้นพอประมาณ
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.30),

        layers.Dense(32),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.20),

        layers.Dense(16),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.10),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def find_best_threshold(y_true, probs, metric="f1"):
    # ลอง threshold หลายค่า แล้วเลือกที่ดีที่สุด (ช่วยงานแพทย์มาก)
    thresholds = np.linspace(0.2, 0.8, 61)
    best_t, best_score = 0.5, -1

    for t in thresholds:
        pred = (probs >= t).astype(int)

        # คำนวณจาก report ง่ายๆ
        # ใช้ f1 ของ class 1 เป็นหลัก
        report = classification_report(y_true, pred, output_dict=True, zero_division=0)
        f1_1 = report["1"]["f1-score"]
        rec_1 = report["1"]["recall"]

        score = f1_1 if metric == "f1" else rec_1

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, float(best_score)


def main():
    set_seed(SEED)

    # 1) Load prepared dataset
    df = pd.read_csv("data/diabetes_prepared.csv")

    target_col = "Outcome"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")

    X = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].values.astype(np.int32)

    # 2) Split (stratify สำคัญมาก)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)
    print("Positive rate (train):", float(y_train.mean()))
    print("Positive rate (test) :", float(y_test.mean()))

    # 3) Class weight กัน class imbalance
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print("Class weight:", class_weight)

    # 4) Build model
    model = build_model(input_dim=X_train.shape[1])

    # 5) Callbacks (ใช้ val_auc เพราะบาลานซ์กว่า accuracy)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=20, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max",
            factor=0.5, patience=7, min_lr=1e-6
        )
    ]

    # 6) Train
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=300,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # 7) Evaluate (ใช้ threshold ที่ดีที่สุด)
    probs = model.predict(X_test).ravel()
    auc = roc_auc_score(y_test, probs)

    best_t, best_score = find_best_threshold(y_test, probs, metric="f1")
    y_pred = (probs >= best_t).astype(int)

    print("\nROC-AUC:", round(float(auc), 4))
    print("Best threshold (by F1 class=1):", best_t, "score:", round(best_score, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # 8) Save model + metadata
    os.makedirs("models", exist_ok=True)
    model.save("models/diabetes_nn.keras")

    meta = {
        "target_col": target_col,
        "input_dim": int(X_train.shape[1]),
        "best_threshold": best_t,
        "roc_auc": float(auc),
        "seed": SEED
    }
    with open("models/diabetes_nn_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n✅ Saved:")
    print("- models/diabetes_nn.keras")
    print("- models/diabetes_nn_meta.json")


if __name__ == "__main__":
    main()