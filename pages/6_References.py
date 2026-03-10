import streamlit as st

st.set_page_config(page_title="References", page_icon="📚", layout="wide")

# ---------- Style ----------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}
.hero {
    background: linear-gradient(135deg, #fffdf8 0%, #fff7ed 100%);
    padding: 28px;
    border-radius: 20px;
    border: 1px solid #fde7c7;
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
    <h1 style="margin-bottom:8px;">📚 References</h1>
    <div class="small-muted">
        This page lists the datasets, libraries, and technical resources used in this project.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Datasets ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("1. Dataset References")
st.markdown("""
### Diabetes Dataset
- **Pima Indians Diabetes Database**
- Source: UCI Machine Learning Repository / Kaggle mirror
- Common link:  
  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### Heart Disease Dataset
- **Heart Disease Dataset**
- Source: UCI Machine Learning Repository
- Common link:  
  https://archive.ics.uci.edu/ml/datasets/heart+disease
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Machine Learning ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("2. Machine Learning References")
st.markdown("""
- **Scikit-learn Documentation**  
  https://scikit-learn.org/stable/

- **Logistic Regression**  
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

- **Random Forest Classifier**  
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

- **Gradient Boosting Classifier**  
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

- **Voting Classifier / Ensemble Methods**  
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Deep Learning ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("3. Neural Network References")
st.markdown("""
- **TensorFlow Documentation**  
  https://www.tensorflow.org/

- **Keras Documentation**  
  https://keras.io/

- **Dense Layer**  
  https://keras.io/api/layers/core_layers/dense/

- **Dropout Layer**  
  https://keras.io/api/layers/regularization_layers/dropout/

- **Batch Normalization**  
  https://keras.io/api/layers/normalization_layers/batch_normalization/

- **EarlyStopping Callback**  
  https://keras.io/api/callbacks/early_stopping/

- **ReduceLROnPlateau Callback**  
  https://keras.io/api/callbacks/reduce_lr_on_plateau/
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Evaluation ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("4. Evaluation Metrics References")
st.markdown("""
- **Confusion Matrix**  
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

- **Classification Report**  
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

- **ROC Curve**  
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html

- **ROC-AUC Score**  
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

- **Precision, Recall, F1-score**  
  https://scikit-learn.org/stable/modules/model_evaluation.html
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Web / Deployment ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("5. Web Application and Deployment References")
st.markdown("""
- **Streamlit Documentation**  
  https://docs.streamlit.io/

- **Streamlit Community Cloud**  
  https://streamlit.io/cloud

- **Matplotlib Documentation**  
  https://matplotlib.org/

- **Pandas Documentation**  
  https://pandas.pydata.org/

- **NumPy Documentation**  
  https://numpy.org/doc/
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Note ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("6. Note")
st.write("""
This project is developed for **educational purposes** in the Intelligent System course.
Some implementation support, code refinement, and explanation drafting were assisted by AI tools during development.
""")
st.markdown('</div>', unsafe_allow_html=True)

st.success("References page completed successfully.")