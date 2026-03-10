import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="ML Approach", page_icon="🧠", layout="wide")

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
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1 style="margin-bottom:8px;">🧠 Machine Learning Approach</h1>
    <div class="small-muted">
        This page explains the complete development workflow of the Machine Learning model,
        from data preparation to model training, evaluation, and deployment.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Overview ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("1. Project Objective")
st.write("""
The objective of this project is to develop a **disease risk prediction system**
for **Diabetes** and **Heart Disease** using **Machine Learning ensemble methods**.

The system is designed to:
- process imperfect medical datasets,
- train predictive models,
- compare model performance,
- and provide an interactive web-based prediction interface.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Dataset ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("2. Datasets Used")
st.write("""
This project uses two datasets:

### 2.1 Diabetes Dataset
- **Name:** Pima Indians Diabetes Dataset
- **Source:** UCI / Kaggle
- **Target column:** `Outcome`
- **Task:** Binary classification (0 = No diabetes, 1 = Diabetes)

### 2.2 Heart Disease Dataset
- **Name:** Heart Disease UCI Dataset
- **Source:** UCI Machine Learning Repository
- **Target column:** `num`
- **Task:** Binary classification (0 = No heart disease, 1 = Heart disease)
""")

st.write("Both datasets contain **missing values, inconsistent values, or imperfect records**, which makes data preparation necessary before model training.")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Preprocessing ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("3. Data Preparation")
st.write("""
Since the datasets are not perfectly clean, the following data preparation steps were performed:

### 3.1 Data Cleaning
- Removed invalid or inconsistent values
- Replaced missing values with appropriate estimates
- Checked data types and corrected numeric conversion issues

### 3.2 Feature Encoding
- Converted categorical values into numeric format
- Converted boolean/text features into machine-readable form

### 3.3 Feature Scaling
- Applied **StandardScaler** to normalize the input features
- This helps the models learn more effectively and improves stability

### 3.4 Train-Test Split
- The prepared data was split into:
  - **80% training set**
  - **20% testing set**
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Algorithm Theory ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("4. Machine Learning Theory")
st.write("""
To improve performance and robustness, an **Ensemble Machine Learning model** was used.

### Base Models Included
1. **Logistic Regression**
   - A linear classification algorithm
   - Useful as a baseline model
   - Interpretable and efficient

2. **Random Forest**
   - A bagging-based ensemble of decision trees
   - Handles non-linear relationships well
   - Reduces overfitting compared to a single tree

3. **Gradient Boosting**
   - A boosting-based method
   - Builds trees sequentially to reduce previous errors
   - Often achieves strong predictive performance

### Ensemble Strategy
The final Machine Learning model uses **Soft Voting Ensemble**:
- Each base model predicts a probability
- The probabilities are averaged
- The final class is selected based on a decision threshold

This approach improves generalization by combining the strengths of multiple algorithms.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Development Procedure ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("5. Model Development Procedure")
st.write("""
The development process of the Machine Learning model is summarized below:

1. Load the prepared dataset  
2. Separate features (`X`) and target (`y`)  
3. Split the data into training and testing sets  
4. Train the following models:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
5. Combine them using **VotingClassifier**
6. Evaluate model performance on the testing set
7. Save the trained models for deployment in the Streamlit application
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Workflow Overview ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("6. Ensemble Workflow Overview")

workflow_df = pd.DataFrame({
    "Stage": [1, 2, 3, 4, 5, 6, 7],
    "Step": [
        "Load Prepared Data",
        "Split Features and Target",
        "Train Logistic Regression",
        "Train Random Forest",
        "Train Gradient Boosting",
        "Combine with Soft Voting",
        "Evaluate and Save Model"
    ],
    "Y": [1, 1, 1, 1, 1, 1, 1]
})

fig = px.line(
    workflow_df,
    x="Stage",
    y="Y",
    markers=True,
    text="Step",
    title="Machine Learning Development Workflow"
)

fig.update_traces(
    textposition="top center",
    line=dict(width=3),
    marker=dict(size=16)
)

fig.update_layout(
    showlegend=False,
    xaxis_title="Development Stage",
    yaxis_title="",
    height=450,
    yaxis=dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        range=[0.8, 1.2]
    ),
    xaxis=dict(
        tickmode="array",
        tickvals=[1, 2, 3, 4, 5, 6, 7]
    )
)

st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Evaluation ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("7. Evaluation Metrics")
st.write("""
The Machine Learning model was evaluated using several classification metrics:

- **Accuracy**: overall correctness of predictions
- **Precision**: how many predicted positive cases are truly positive
- **Recall**: how many actual positive cases are correctly detected
- **F1-score**: harmonic mean of precision and recall
- **ROC-AUC**: ability of the model to separate classes
- **Confusion Matrix**: detailed analysis of true/false predictions

These metrics are important because in medical applications,
**accuracy alone is not enough**.
For example, a model with high accuracy may still fail to detect positive patients.
Therefore, recall and F1-score are also emphasized.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Deployment ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("8. Deployment")
st.write("""
After training, the Machine Learning ensemble model was saved as a `.joblib` file
and integrated into the Streamlit web application.

The deployed system allows users to:
- input medical features,
- run prediction in real time,
- choose a decision threshold,
- and view the final risk prediction.

This web application can be deployed to **Streamlit Community Cloud**
or other cloud platforms.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- References ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("9. References")
st.markdown("""
- UCI Machine Learning Repository  
- Kaggle Datasets  
- Scikit-learn Documentation  
- Ensemble Learning Methods in Machine Learning  
- Streamlit Documentation  
""")
st.markdown('</div>', unsafe_allow_html=True)

st.success("Machine Learning explanation page completed successfully.")