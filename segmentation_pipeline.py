import joblib
from src.components.segmentation import ModelCluster
from src.components.DataIngestion import DataIngestion
from src.components.DataPreprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
class SegmentModel:
    def run_segment_model(self):

        ingestion = DataIngestion()
        data = ingestion.ingest_data(r"C:\Users\DELL\OneDrive\Desktop\Customer_Churn_Prediction\Customer-Churn-Prediction-and-Segmentation\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

        print("Data Ingestion Complete!")

        model_pipeline = joblib.load("models/model.pkl")
        preprocessor = model_pipeline.named_steps["preprocessor"]
        
        print("Loaded trained churn model!")
        
        cluster=ModelCluster()
        data,kmeans,preprocessor=cluster.model_segment(data,preprocessor,model_pipeline)
        
        print("Segmentation done")

        
if __name__=="__main__":
    segmentation=SegmentModel()
    segmentation.run_segment_model()