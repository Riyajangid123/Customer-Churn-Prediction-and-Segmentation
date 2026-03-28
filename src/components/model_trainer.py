
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.model_selection import GridSearchCV


class ModelTrainer:

    def TrainModel(self, x, y, preprocessor):

        # train test split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        # full pipeline (preprocessing + model)
        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model",XGBClassifier(scale_pos_weight=2.77,random_state=42,max_depth=1,learning_rate=1.0))
            ]
        )


        model_pipeline.fit(x_train, y_train)

        
        y_pred = model_pipeline.predict(x_test)

    
        precision= precision_score(y_test, y_pred)
        recall=recall_score(y_test,y_pred)

        print("Model precision:", precision)
        print("Model recall:",recall)
        fold=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        cross=cross_val_score(model_pipeline,x,y,cv=fold,scoring="roc_auc")

        print("cross_validation_score:",cross.mean())
        
        
        os.makedirs("models", exist_ok=True)

        # save model
        joblib.dump(model_pipeline, "models/model.pkl")

        print("Model saved successfully!")

        return model_pipeline
