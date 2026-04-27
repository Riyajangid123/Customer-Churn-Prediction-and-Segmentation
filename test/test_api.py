from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app

client = TestClient(app)

sample_data = {
  "customerID": "1001-ABCD",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 18,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.60,
  "TotalCharges": 1540.80,
}

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Churn Prediction API running"}


def test_predict_response_structure():
    response = client.post("/predict_full", json=sample_data)
    assert response.status_code == 200
    json_data = response.json()

    assert "segment" in json_data
    assert "churn_prob" in json_data


def test_predict_segment_values():
    response = client.post("/predict_full", json=sample_data)
    assert response.json()["segment"] in ["High Value", "Low Value", "At Risk"]


def test_predict_missing_fields():
    response = client.post("/predict_full", json={"TotalCharges": 1500.0})
    assert response.status_code == 422


def test_churn_probability_range():
    response = client.post("/predict_full", json=sample_data)
    prob = response.json()["churn_prob"]
    assert 0.0 <= prob <= 1.0