import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,roc_auc_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
class ModelTrainer:
    def train_model(self,x,y,preprocessor):

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
        scale_pos_weight=5174/1869
        
        model_pipeline=Pipeline([("preprocessor",preprocessor),
                                       ("model",XGBClassifier(scale_pos_weight=2.77,learning_rate=1.0,max_depth=1,
                                                                   random_state=42))])
                                               
        model_pipeline.fit(x_train,y_train)

        y_pred=model_pipeline.predict(x_test)
        precision= precision_score(y_test, y_pred)
        recall=recall_score(y_test,y_pred)

        print(f"precision-> {precision}")
        print(f"Recall-> {recall}")

        fold=KFold(n_splits=5,shuffle=True,random_state=42)
        score=cross_val_score(model_pipeline,x,y,cv=fold,scoring="roc_auc")
        print(f"Cross_Val_Score-> {score.mean()}")
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(model_pipeline, "models/model.pkl")

        print("model saved successfully")

        return model_pipeline

