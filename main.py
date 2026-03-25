import joblib
import pandas as pd
from fastapi import FastAPI
from schema.schema import CustomerChurn

app=FastAPI()

model=joblib.load("models/model.pkl")
segment_model = joblib.load("models/kmeans.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

@app.get("/")
def home():
    return {"message":"Churn Prediction API running"}

@app.post("/predict_full")
def predict_full(data: CustomerChurn):

    input_dict = data.model_dump()
    input_data = pd.DataFrame([input_dict])
    input_data = input_data.drop(columns=["customerID"], errors="ignore")

    # Churn
    churn_pred = model.predict(input_data)[0]

    # Segmentation
    processed = preprocessor.transform(input_data)
    segment = segment_model.predict(processed)[0]

    segment_map = {
        0: "High Value",
        1: "Low Value",
        2: "At Risk"
    }

    return {
        "churn": "Yes" if churn_pred == 1 else "No",
        "segment": segment_map[segment]
    }