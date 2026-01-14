import pickle
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from CONFIG import features, directories

#### Variables ####

# Directories
log_dir = directories["log_dir"]
model_dir = directories["model_dir"]

# Features
categorical_variables = features["categorical_variables"]
numerical_variables = features["numerical_variables"]
binary_variables = features["binary_variables"]

#### Loading Files ####

# Loading pkl files
with open("one_hot_encoder.pkl", "rb") as f:
    one_hot_encoder = pickle.load(f)

with open("min_max_scaler.pkl", "rb") as f:
    min_max_scaler = pickle.load(f)

# Loading model
model = load_model(f"{model_dir}/model_1.keras")

#### Streamlit app ####

st.title("Customer Staying Prediction")
st.text("This tool takes some inputs and predicts if the customer will stay or not.")

# User inputs
geography = st.selectbox("Geography", one_hot_encoder.categories_[0])
gender = st.selectbox("Gender", one_hot_encoder.categories_[1])
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])


# Preparing Data
input_df = pd.DataFrame(
    {
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary],
    }
)

categorical_encoded = pd.DataFrame(
    one_hot_encoder.transform(input_df[categorical_variables])
)
numerical_standardized = pd.DataFrame(
    min_max_scaler.transform(input_df[numerical_variables])
)
preprocessed_df = pd.concat(
    [
        input_df[binary_variables].reset_index(drop=True),
        categorical_encoded,
        numerical_standardized,
    ],
    axis=1,
)

# Predicting and Displaying prediction
prediction_prob = model.predict(preprocessed_df)

if prediction_prob > 0.5:
    st.write("### **The customer is likely to exit.**")
else:
    st.write("### **The customer is likely to stay.**")

st.write(f"### **Stay Probability: {(1-prediction_prob[0][0]):.2f}**")
