import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction System")

st.write("Enter Customer Details")

credit_score = st.number_input("Credit Score")
age = st.number_input("Age")
balance = st.number_input("Balance")
products = st.number_input("Number of Products")
salary = st.number_input("Estimated Salary")

if st.button("Predict Churn"):

    data = np.array([[credit_score, age, balance, products, salary]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    if prediction > 0.5:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer will stay")
