import pandas as pd

class DataIngestion:
    def ingest_data(self,path):
        data=pd.read_csv(path)
        return data