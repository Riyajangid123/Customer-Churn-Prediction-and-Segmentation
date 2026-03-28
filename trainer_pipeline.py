from src.components.DataIngestion import DataIngestion
from src.components.DataPreprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def run_pipeline(self):
        ingestion=DataIngestion()

        data=ingestion.ingest_data(r"C:\Users\DELL\OneDrive\Desktop\Customer_Churn_Prediction\Customer-Churn-Prediction-and-Segmentation\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

        print("Data Ingestion done")

        preprocessing=DataPreprocessing()

        x,y,preprocessor=preprocessing.data_preprocess(data)

        print("Data Preprocessing done")

        trainer = ModelTrainer()
        trainer.TrainModel(x, y, preprocessor)

        print("Model Training done")

if __name__=="__main__":

    pipeline=TrainPipeline()
    pipeline.run_pipeline()
