import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Test ML Model", page_icon="🧪", layout="wide")

# ---------- Style ----------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}
.hero {
    background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
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
.result-good {
    background: #ecfdf3;
    color: #166534;
    border: 1px solid #bbf7d0;
    padding: 14px;
    border-radius: 14px;
    font-weight: 600;
}
.result-bad {
    background: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
    padding: 14px;
    border-radius: 14px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
@st.cache_data
def load_data():
    diabetes_df = pd.read_csv("data/diabetes_prepared.csv")
    heart_df = pd.read_csv("data/heart_prepared.csv")
    return diabetes_df, heart_df

@st.cache_resource
def load_models():
    diabetes_model = joblib.load("models/diabetes_ensemble_voting.joblib")
    heart_model = joblib.load("models/heart_ensemble_voting.joblib")
    return diabetes_model, heart_model

def predict_probability(model, x_input):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(x_input)[0, 1])
    pred = float(model.predict(x_input)[0])
    return pred

def render_result(probability, threshold, positive_label):
    prediction = 1 if probability >= threshold else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Probability", f"{probability:.4f}")
    with c2:
        st.metric("Threshold", f"{threshold:.2f}")
    with c3:
        st.metric("Prediction", f"{prediction}")

    st.write("")
    if prediction == 1:
        st.markdown(
            f"<div class='result-bad'>⚠️ Predicted: {positive_label} (High Risk)</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-good'>✅ Predicted: Negative (Low Risk)</div>",
            unsafe_allow_html=True
        )

    st.caption("Disclaimer: This prediction is for educational purposes only and is not a medical diagnosis.")

# ---------- Load ----------
diabetes_df, heart_df = load_data()
diabetes_model, heart_model = load_models()

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1 style="margin-bottom:8px;">🧪 Test Machine Learning Model</h1>
    <div class="small-muted">
        This page allows users to test the Ensemble Machine Learning model
        for Diabetes and Heart Disease prediction.
    </div>
</div>
""", unsafe_allow_html=True)

st.write("The input values below are based on the **prepared/scaled datasets** used during training.")

tab1, tab2 = st.tabs(["Diabetes (ML)", "Heart Disease (ML)"])

# =========================================================
# TAB 1 — DIABETES
# =========================================================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Diabetes Prediction using Ensemble Machine Learning")
    st.caption("These features are preprocessed values consistent with the training pipeline.")

    diabetes_features = [col for col in diabetes_df.columns if col != "Outcome"]

    col_a, col_b = st.columns([2, 1])

    with col_b:
        st.markdown("**Quick Fill Example**")
        example_idx = st.number_input(
            "Select sample row",
            min_value=0,
            max_value=len(diabetes_df)-1,
            value=0,
            step=1,
            key="diab_idx"
        )
        threshold = st.slider(
            "Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.01,
            key="diab_threshold"
        )
        default_values = diabetes_df.iloc[int(example_idx)][diabetes_features].to_dict()

    with col_a:
        st.markdown("**Input Features**")
        cols = st.columns(2)
        diabetes_inputs = {}
        for i, feature in enumerate(diabetes_features):
            with cols[i % 2]:
                diabetes_inputs[feature] = st.number_input(
                    feature,
                    value=float(default_values.get(feature, 0.0)),
                    step=0.1,
                    format="%.4f",
                    key=f"diab_{feature}"
                )

    if st.button("Predict Diabetes (ML)", use_container_width=True):
        x = np.array([diabetes_inputs[f] for f in diabetes_features], dtype=float).reshape(1, -1)
        probability = predict_probability(diabetes_model, x)
        st.write("")
        render_result(probability, threshold, "Diabetes")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 2 — HEART
# =========================================================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Heart Disease Prediction using Ensemble Machine Learning")
    st.caption("These features are preprocessed values consistent with the training pipeline.")

    heart_features = [col for col in heart_df.columns if col != "num"]

    col_a, col_b = st.columns([2, 1])

    with col_b:
        st.markdown("**Quick Fill Example**")
        example_idx_h = st.number_input(
            "Select sample row",
            min_value=0,
            max_value=len(heart_df)-1,
            value=0,
            step=1,
            key="heart_idx"
        )
        threshold_h = st.slider(
            "Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.40,
            step=0.01,
            key="heart_threshold"
        )
        default_values_h = heart_df.iloc[int(example_idx_h)][heart_features].to_dict()

    with col_a:
        st.markdown("**Input Features**")
        cols_h = st.columns(3)
        heart_inputs = {}
        for i, feature in enumerate(heart_features):
            with cols_h[i % 3]:
                heart_inputs[feature] = st.number_input(
                    feature,
                    value=float(default_values_h.get(feature, 0.0)),
                    step=0.1,
                    format="%.4f",
                    key=f"heart_{feature}"
                )

    if st.button("Predict Heart Disease (ML)", use_container_width=True):
        x = np.array([heart_inputs[f] for f in heart_features], dtype=float).reshape(1, -1)
        probability = predict_probability(heart_model, x)
        st.write("")
        render_result(probability, threshold_h, "Heart Disease")

    st.markdown('</div>', unsafe_allow_html=True)