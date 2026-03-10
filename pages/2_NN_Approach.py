import streamlit as st

st.set_page_config(page_title="NN Approach", page_icon="🧠", layout="wide")

# ---------- Style ----------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}
.hero {
    background: linear-gradient(135deg, #fff8fb 0%, #f5f0ff 100%);
    padding: 28px;
    border-radius: 20px;
    border: 1px solid #eadcf8;
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
.arch-box {
    background: #faf7ff;
    border: 1px solid #eadcf8;
    border-radius: 14px;
    padding: 16px;
    font-family: monospace;
    line-height: 1.8;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1 style="margin-bottom:8px;">🧠 Neural Network Approach</h1>
    <div class="small-muted">
        This page explains the complete development workflow of the Neural Network model,
        including data preparation, deep learning theory, model architecture,
        training procedure, evaluation, and deployment.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Introduction ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("1. Project Objective")
st.write("""
The objective of the Neural Network model is to predict the risk of **Diabetes**
and **Heart Disease** from medical input features.

Unlike traditional linear models, Neural Networks are capable of learning
**complex nonlinear relationships** between variables.
This makes them suitable for healthcare prediction tasks,
especially when multiple clinical features interact with each other.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Dataset ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("2. Datasets Used")
st.write("""
The same two datasets were used for the Neural Network model:

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

st.write("""
These datasets contain **missing, inconsistent, and imperfect values**,
therefore data preparation is necessary before feeding them into the neural network.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Data Preparation ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("3. Data Preparation")
st.write("""
The Neural Network requires numeric and well-scaled input features.
Therefore, the following preprocessing steps were applied:

### 3.1 Data Cleaning
- Replaced invalid values such as `"?"`, `"NA"`, and blank strings with missing values
- Converted corrupted numeric values into proper numeric format
- Removed unnecessary or duplicated columns when needed

### 3.2 Missing Value Handling
- Missing numeric values were filled using **median imputation**
- Median was chosen because it is robust against outliers

### 3.3 Encoding Categorical Features
- Text-based categories were transformed into numeric values
- Boolean features such as TRUE/FALSE were converted into 1/0

### 3.4 Feature Scaling
- Applied **StandardScaler**
- Neural Networks perform better when input features are normalized
- This helps gradient-based learning converge more efficiently

### 3.5 Train-Test Split
- Data was split into:
  - **80% training set**
  - **20% testing set**
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Theory ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("4. Neural Network Theory")
st.write("""
A Neural Network is a computational model inspired by the structure of the human brain.
It consists of layers of neurons that transform input data into predictions.

### Key Idea
Each neuron computes:
- weighted sum of inputs
- adds a bias
- passes the result through an activation function

Mathematically:

**z = w·x + b**

Then the output is passed to an activation function.

### Why Neural Networks?
Neural Networks are useful because they can:
- learn nonlinear patterns
- model hidden interactions between variables
- improve performance when the relationship between features is complex
""")

st.write("""
### Activation Function
In this project, the **ReLU (Rectified Linear Unit)** activation function is used in hidden layers:

**ReLU(x) = max(0, x)**

Advantages of ReLU:
- simple and fast
- reduces vanishing gradient problems
- widely used in deep learning
""")

st.write("""
### Output Layer
For binary classification, the output layer uses **Sigmoid activation**:

**Sigmoid(x) = 1 / (1 + e^(-x))**

This produces a value between **0 and 1**, interpreted as probability.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Architecture ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("5. Model Architecture")

st.write("""
The Neural Network architecture was designed specifically for tabular medical data.

Main components:
- Dense layers for feature learning
- Batch Normalization for training stability
- Dropout for regularization
- Sigmoid output layer for binary prediction
""")

st.markdown("""
<div class="arch-box">
Input Layer<br>
↓<br>
Dense Layer (64 neurons)<br>
↓<br>
Batch Normalization<br>
↓<br>
ReLU Activation<br>
↓<br>
Dropout (0.30)<br>
↓<br>
Dense Layer (32 neurons)<br>
↓<br>
Batch Normalization<br>
↓<br>
ReLU Activation<br>
↓<br>
Dropout (0.20)<br>
↓<br>
Output Layer (1 neuron, Sigmoid)
</div>
""", unsafe_allow_html=True)

st.write("""
### Why this architecture?
- **Dense(64, 32)** provides enough capacity for learning patterns without being too large
- **Batch Normalization** improves convergence and stabilizes internal distributions
- **Dropout** helps reduce overfitting
- **Sigmoid output** is appropriate for binary classification
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Training ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("6. Model Training Procedure")
st.write("""
The Neural Network was trained using the following workflow:

1. Load the prepared dataset  
2. Separate input features (`X`) and target labels (`y`)  
3. Split the dataset into training and testing sets  
4. Compute **class weights** to handle imbalance in positive and negative samples  
5. Build the neural network architecture  
6. Compile the model with:
   - **Optimizer:** Adam
   - **Loss function:** Binary Crossentropy
   - **Metrics:** Accuracy, Precision, Recall, AUC
7. Train the model on the training data  
8. Validate performance on validation data  
9. Save the trained model for web deployment
""")

st.write("""
### Training Optimization Techniques
To improve learning and prevent overfitting, the following techniques were used:

- **EarlyStopping**
  - stops training when validation performance no longer improves

- **ReduceLROnPlateau**
  - reduces learning rate automatically when improvement slows down

- **Class Weighting**
  - helps the model learn better when the dataset is imbalanced
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Evaluation ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("7. Model Evaluation")
st.write("""
The Neural Network model was evaluated using several classification metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC**
- **Confusion Matrix**

### Why multiple metrics are important
In healthcare prediction tasks, relying only on accuracy can be misleading.
For example, a model may predict the majority class correctly most of the time
but still fail to identify actual positive patients.

Therefore, this project also emphasizes:
- **Recall** → ability to detect positive cases
- **F1-score** → balance between precision and recall
- **ROC-AUC** → quality of probability ranking
""")

st.write("""
### Threshold Adjustment
The model produces a probability score between 0 and 1.

A threshold is then used to convert probability into a class label:
- Probability ≥ threshold → Positive
- Probability < threshold → Negative

This threshold can be adjusted in the web application
to control model sensitivity depending on the use case.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Deployment ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("8. Deployment")
st.write("""
After training, the Neural Network model was saved in **Keras format (`.keras`)**
and integrated into the Streamlit web application.

The deployed system allows users to:
- enter medical values,
- run prediction interactively,
- view probability scores,
- adjust thresholds,
- and compare Neural Network performance with Machine Learning models.

This deployment makes the model accessible through a user-friendly web interface.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Comparison ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("9. Comparison with Machine Learning")
st.write("""
The Neural Network model was developed as a second modeling approach
to compare against the Ensemble Machine Learning model.

### Comparison goals:
- determine whether deep learning improves prediction quality
- compare generalization ability on tabular data
- study trade-offs between interpretability and predictive power

In this project:
- **Machine Learning Ensemble** provides strong baseline performance
- **Neural Network** provides a nonlinear deep learning alternative
- both are evaluated and compared in the web application
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- References ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("10. References")
st.markdown("""
- TensorFlow / Keras Documentation  
- UCI Machine Learning Repository  
- Kaggle Datasets  
- Deep Learning literature on MLP, Batch Normalization, and Dropout  
- Streamlit Documentation  
""")
st.markdown('</div>', unsafe_allow_html=True)

st.success("Neural Network explanation page completed successfully.")
