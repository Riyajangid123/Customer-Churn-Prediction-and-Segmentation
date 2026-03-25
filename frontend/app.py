import streamlit as st
import requests

st.set_page_config(page_title="Customer Intelligence System", layout="centered")

st.title("Customer Intelligence System")
st.write("Predict Customer Churn & Segment")
st.markdown("---")

# ========== INPUT FIELDS ==========
customerID = st.text_input("Customer ID")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner?", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

tenure = st.number_input("Tenure (months)", min_value=0)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input("Monthly Charges")
TotalCharges = st.number_input("Total Charges")

st.markdown("---")

# ========== BUTTON ==========
if st.button("Analyze Customer"):

    url = "http://127.0.0.1:8000/predict_full"

    payload = {
        "customerID": customerID,
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()

        st.markdown("## Results")

        # Churn result
        if result["churn"] == "Yes":
            st.error("High Risk: Customer likely to churn")
        else:
            st.success("Low Risk: Customer likely to stay")

        # Segment result
        st.info(f"Customer Segment: {result['segment']}")

    else:
        st.error("Error connecting to API")