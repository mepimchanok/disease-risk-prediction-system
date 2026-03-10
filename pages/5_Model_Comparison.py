import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")

# ---------- Style ----------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1150px;
}
.hero {
    background: linear-gradient(135deg, #f8fbff 0%, #f4f7ff 100%);
    padding: 28px;
    border-radius: 20px;
    border: 1px solid #dbe7ff;
    margin-bottom: 20px;
}
.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #ececec;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
    margin-bottom: 18px;
}
.small-muted {
    color: #6b7280;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- TensorFlow ----------
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf = None
    keras = None

# ---------- Helpers ----------
@st.cache_data
def load_data():
    diabetes_df = pd.read_csv("data/diabetes_prepared.csv")
    heart_df = pd.read_csv("data/heart_prepared.csv")
    return diabetes_df, heart_df

@st.cache_resource
def load_models():
    diabetes_ml = joblib.load("models/diabetes_ensemble_voting.joblib")
    heart_ml = joblib.load("models/heart_ensemble_voting.joblib")

    diabetes_nn = None
    heart_nn = None
    if keras is not None:
        diabetes_nn = keras.models.load_model("models/diabetes_nn.keras")
        heart_nn = keras.models.load_model("models/heart_nn.keras")

    return diabetes_ml, heart_ml, diabetes_nn, heart_nn

def get_xy(df, target_col):
    X = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].astype(int).values
    return X, y

def predict_proba_sklearn(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return pred.astype(float)

def predict_proba_nn(model, X):
    return model.predict(X, verbose=0).ravel().astype(float)

def evaluate_model(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "y_pred": y_pred
    }

def plot_metric_bar(df_metrics, title):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metric_names))
    width = 0.35

    ml_vals = [df_metrics.loc[df_metrics["Model"] == "Ensemble ML", m].values[0] for m in metric_names]
    nn_vals = [df_metrics.loc[df_metrics["Model"] == "Neural Network", m].values[0] for m in metric_names]

    ax.bar(x - width/2, ml_vals, width, label="Ensemble ML")
    ax.bar(x + width/2, nn_vals, width, label="Neural Network")

    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, proba_ml, proba_nn, title):
    fpr_ml, tpr_ml, _ = roc_curve(y_true, proba_ml)
    fpr_nn, tpr_nn, _ = roc_curve(y_true, proba_nn)

    auc_ml = roc_auc_score(y_true, proba_ml)
    auc_nn = roc_auc_score(y_true, proba_nn)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_ml, tpr_ml, label=f"Ensemble ML (AUC={auc_ml:.3f})")
    ax.plot(fpr_nn, tpr_nn, label=f"Neural Network (AUC={auc_nn:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_confusion(cm, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")

    plt.tight_layout()
    return fig

# ---------- Load ----------
diabetes_df, heart_df = load_data()
diabetes_ml, heart_ml, diabetes_nn, heart_nn = load_models()

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1 style="margin-bottom:8px;">📊 Model Comparison</h1>
    <div class="small-muted">
        Compare Ensemble Machine Learning and Neural Network models
        using multiple evaluation metrics and visualizations.
    </div>
</div>
""", unsafe_allow_html=True)

if keras is None:
    st.error("TensorFlow / Keras is not installed. Neural Network comparison cannot be loaded.")
    st.stop()

tab1, tab2 = st.tabs(["Diabetes Comparison", "Heart Disease Comparison"])

# =========================================================
# TAB 1 — DIABETES
# =========================================================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Diabetes Model Comparison")

    X_d, y_d = get_xy(diabetes_df, "Outcome")
    threshold_d = st.slider("Diabetes Threshold", 0.05, 0.95, 0.50, 0.01, key="th_d")

    proba_ml_d = predict_proba_sklearn(diabetes_ml, X_d)
    proba_nn_d = predict_proba_nn(diabetes_nn, X_d)

    eval_ml_d = evaluate_model(y_d, proba_ml_d, threshold_d)
    eval_nn_d = evaluate_model(y_d, proba_nn_d, threshold_d)

    metrics_df_d = pd.DataFrame([
        {
            "Model": "Ensemble ML",
            "accuracy": eval_ml_d["accuracy"],
            "precision": eval_ml_d["precision"],
            "recall": eval_ml_d["recall"],
            "f1": eval_ml_d["f1"],
            "roc_auc": eval_ml_d["roc_auc"]
        },
        {
            "Model": "Neural Network",
            "accuracy": eval_nn_d["accuracy"],
            "precision": eval_nn_d["precision"],
            "recall": eval_nn_d["recall"],
            "f1": eval_nn_d["f1"],
            "roc_auc": eval_nn_d["roc_auc"]
        }
    ])

    st.markdown("### Performance Table")
    st.dataframe(
        metrics_df_d.style.format({
            "accuracy": "{:.4f}",
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1": "{:.4f}",
            "roc_auc": "{:.4f}"
        }),
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Metric Comparison")
        st.pyplot(plot_metric_bar(metrics_df_d, "Diabetes Model Metrics"))
    with col2:
        st.markdown("### ROC Curve")
        st.pyplot(plot_roc_curve(y_d, proba_ml_d, proba_nn_d, "Diabetes ROC Curve"))

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Confusion Matrix — Ensemble ML")
        cm_ml_d = confusion_matrix(y_d, eval_ml_d["y_pred"])
        st.pyplot(plot_confusion(cm_ml_d, "Diabetes ML"))
    with col4:
        st.markdown("### Confusion Matrix — Neural Network")
        cm_nn_d = confusion_matrix(y_d, eval_nn_d["y_pred"])
        st.pyplot(plot_confusion(cm_nn_d, "Diabetes NN"))

    better_model_d = "Ensemble ML" if eval_ml_d["f1"] >= eval_nn_d["f1"] else "Neural Network"
    st.success(f"Suggested model for Diabetes (based on F1-score): **{better_model_d}**")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 2 — HEART
# =========================================================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Heart Disease Model Comparison")

    X_h, y_h = get_xy(heart_df, "num")
    threshold_h = st.slider("Heart Disease Threshold", 0.05, 0.95, 0.40, 0.01, key="th_h")

    proba_ml_h = predict_proba_sklearn(heart_ml, X_h)
    proba_nn_h = predict_proba_nn(heart_nn, X_h)

    eval_ml_h = evaluate_model(y_h, proba_ml_h, threshold_h)
    eval_nn_h = evaluate_model(y_h, proba_nn_h, threshold_h)

    metrics_df_h = pd.DataFrame([
        {
            "Model": "Ensemble ML",
            "accuracy": eval_ml_h["accuracy"],
            "precision": eval_ml_h["precision"],
            "recall": eval_ml_h["recall"],
            "f1": eval_ml_h["f1"],
            "roc_auc": eval_ml_h["roc_auc"]
        },
        {
            "Model": "Neural Network",
            "accuracy": eval_nn_h["accuracy"],
            "precision": eval_nn_h["precision"],
            "recall": eval_nn_h["recall"],
            "f1": eval_nn_h["f1"],
            "roc_auc": eval_nn_h["roc_auc"]
        }
    ])

    st.markdown("### Performance Table")
    st.dataframe(
        metrics_df_h.style.format({
            "accuracy": "{:.4f}",
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1": "{:.4f}",
            "roc_auc": "{:.4f}"
        }),
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Metric Comparison")
        st.pyplot(plot_metric_bar(metrics_df_h, "Heart Disease Model Metrics"))
    with col2:
        st.markdown("### ROC Curve")
        st.pyplot(plot_roc_curve(y_h, proba_ml_h, proba_nn_h, "Heart Disease ROC Curve"))

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Confusion Matrix — Ensemble ML")
        cm_ml_h = confusion_matrix(y_h, eval_ml_h["y_pred"])
        st.pyplot(plot_confusion(cm_ml_h, "Heart ML"))
    with col4:
        st.markdown("### Confusion Matrix — Neural Network")
        cm_nn_h = confusion_matrix(y_h, eval_nn_h["y_pred"])
        st.pyplot(plot_confusion(cm_nn_h, "Heart NN"))

    better_model_h = "Ensemble ML" if eval_ml_h["f1"] >= eval_nn_h["f1"] else "Neural Network"
    st.success(f"Suggested model for Heart Disease (based on F1-score): **{better_model_h}**")
    st.markdown('</div>', unsafe_allow_html=True)