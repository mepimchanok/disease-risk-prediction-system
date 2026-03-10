import streamlit as st

st.set_page_config(
    page_title="Disease Risk Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Disease Risk Prediction System")

st.markdown("""
This web application predicts the **risk of Diabetes and Heart Disease**
using two types of machine learning approaches.

---

## Machine Learning Models
The following traditional machine learning algorithms are implemented:

• Logistic Regression  
• Random Forest  
• Gradient Boosting  
• Ensemble Voting Classifier  

These models are trained using structured medical datasets and evaluated
using metrics such as **Accuracy, Precision, Recall, F1-score, and ROC-AUC**.

---

## Neural Network Model

A **Feedforward Neural Network (FFNN)** is implemented using
**TensorFlow / Keras** to capture non-linear relationships in the dataset.

The neural network architecture includes:

• Dense layers  
• Batch Normalization  
• Dropout for regularization  
• Adam optimizer  
• Binary Crossentropy loss  

---

## How to Use This Application

Use the **sidebar** to navigate through the system:

1️⃣ **ML Approach** – explanation of machine learning models  
2️⃣ **NN Approach** – explanation of neural network model  
3️⃣ **Test ML** – test prediction using ML models  
4️⃣ **Test NN** – test prediction using neural network  
5️⃣ **Model Comparison** – compare model performance  
6️⃣ **References** – datasets and technical sources  

---

⚠️ **Disclaimer**

This application is for **educational purposes only** and should not be used for medical diagnosis.
""")

st.success("Use the sidebar to navigate through the pages.")