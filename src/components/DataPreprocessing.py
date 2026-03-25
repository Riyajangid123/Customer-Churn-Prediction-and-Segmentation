import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class DataPreprocessing:
    def data_preprocess(self,data):
        data = data.drop(columns=["customerID"])
        x=data.drop("Churn",axis=1)
        y=data["Churn"]
        y=y.map({"Yes":1,"No":0})
        num_cols=x.select_dtypes(exclude="object").columns
        cat_cols=x.select_dtypes(include="object").columns

        num_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),
                                     ("scaler",StandardScaler())])
        
        cat_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                                     ("encoder",OneHotEncoder(handle_unknown="ignore"))])
        
        preprocessor=ColumnTransformer([("num",num_pipeline,num_cols),
                                        ("cat",cat_pipeline,cat_cols)])
        
        return x,y,preprocessor