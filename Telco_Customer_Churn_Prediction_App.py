# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 23:48:30 2025

@author: Oluwaseun Adeyemi
"""

import streamlit as st
import pandas as pd
import joblib

# Load the trained CatBoost pipeline
catboost_pipeline = joblib.load("catboost_pipeline.pkl")

# ========================
# 🌟 Streamlit App Layout
# ========================
st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📊", layout="wide")

# Main App Title (moved to the top)
st.title("📱 Telecom Customer Churn Predictor")

# Introduction
st.markdown("""
Predict if a customer is likely to **churn** or stay with the company.  
This model uses a **CatBoost Classifier** trained on historical telecom customer data. 
""")

# Sidebar for user input
st.sidebar.header("🛠 Enter Customer Details")

def user_input_features():
    gender = st.sidebar.selectbox("👤 Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("👴 Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("❤️ Has Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("👶 Has Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("📆 Tenure (months)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("📞 Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("📞 Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.sidebar.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("🔒 Online Security", ["Yes", "No", "No internet"])
    online_backup = st.sidebar.selectbox("💾 Online Backup", ["Yes", "No", "No internet"])
    device_protection = st.sidebar.selectbox("🛡 Device Protection", ["Yes", "No", "No internet"])
    tech_support = st.sidebar.selectbox("🛠 Tech Support", ["Yes", "No", "No internet"])
    streaming_tv = st.sidebar.selectbox("📺 Streaming TV", ["Yes", "No", "No internet"])
    streaming_movies = st.sidebar.selectbox("🎬 Streaming Movies", ["Yes", "No", "No internet"])
    contract = st.sidebar.selectbox("📄 Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("🧾 Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("💳 Payment Method", ["Electronic check", "Mailed check", 
                                                            "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.sidebar.number_input("💵 Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.sidebar.number_input("💰 Total Charges", min_value=0.0, value=1500.0)

    data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    return pd.DataFrame(data, index=[0])

# Collect user input
input_df = user_input_features()

# Prediction
prediction = catboost_pipeline.predict(input_df)
prediction_proba = catboost_pipeline.predict_proba(input_df)

# ========================
# 🎯 Prediction Results
# ========================
st.subheader("🔮 Prediction")
st.write("✅ No Churn" if prediction[0]==0 else "⚠️ Churn Risk")

st.subheader("📊 Prediction Probability")
st.write(f"✅ No Churn: {prediction_proba[0][0]*100:.2f}%")
st.write(f"⚠️ Churn: {prediction_proba[0][1]*100:.2f}%")

# ========================
# Optional: Display input data
# ========================
st.subheader("📝 Customer Input Data")
st.dataframe(input_df)
