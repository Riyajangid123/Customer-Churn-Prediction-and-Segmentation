import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.components.model_trainer import ModelTrainer
from src.components.DataPreprocessing import DataPreprocessing

class ModelCluster():
    def model_segment(self,data,preprocessor,model_pipeline):
        data["TotalCharges"]=pd.to_numeric(data["TotalCharges"],errors="coerce")
        X_raw = data.drop(columns=["Churn"], errors='ignore')  
        
        X_numeric_for_kmeans = preprocessor.transform(X_raw)
        kmeans=KMeans(n_clusters=3,random_state=42)

        data["Segment"]=kmeans.fit_predict(X_numeric_for_kmeans)

        data["churn_prob"] = model_pipeline.predict_proba(X_raw)[:, 1]



        os.makedirs("models", exist_ok=True)
        joblib.dump(kmeans, "models/kmeans.pkl")
        joblib.dump(preprocessor, "models/preprocessor.pkl")

        os.makedirs("outputs", exist_ok=True)
        data.to_csv("outputs/segmentation_results.csv", index=False)

        print("Models and segmentation results saved successfully in 'models/' and 'outputs/' folders")
    
        
        return data,kmeans,preprocessor