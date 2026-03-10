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
tf.random.set_seed(SEED)
np.random.seed(SEED)

def find_best_threshold(y_true, probs):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.20, 0.81, 0.01):
        y_pred = (probs >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1

def main():
    df = pd.read_csv("data/heart_prepared.csv")
    target_col = "num"

    X = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].values.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),

        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.30),

        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.25),

        layers.Dense(32),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.20),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            "accuracy",
        ]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=6, min_lr=1e-6),
    ]

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    probs = model.predict(X_test).ravel()
    auc = roc_auc_score(y_test, probs)

    best_t, best_f1 = find_best_threshold(y_test, probs)
    y_pred = (probs >= best_t).astype(int)

    print("\nROC-AUC:", round(float(auc), 4))
    print(f"Best threshold (F1 class=1): {best_t:.2f}  score: {best_f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    os.makedirs("models", exist_ok=True)
    model.save("models/heart_nn.keras")

    meta = {
        "target_col": target_col,
        "input_dim": int(X_train.shape[1]),
        "best_threshold": best_t,
        "roc_auc": float(auc),
        "seed": SEED
    }
    with open("models/heart_nn_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n✅ Saved:")
    print("- models/heart_nn.keras")
    print("- models/heart_nn_meta.json")

if __name__ == "__main__":
    main()