
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
import mlflow
import mlflow.sklearn


class ModelTrainer:

    def TrainModel(self, x, y, preprocessor):

        mlflow.set_tracking_uri("file:./mlruns")

        mlflow.set_experiment("Customer Churn Prediction and Segmentation")

        with mlflow.start_run():

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )
            # weight=5179/1869=2.77

            model_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", XGBClassifier(random_state=42, eval_metric='logloss'))
                ])
            
            param_grid = {
                 'model__max_depth': [3, 5, 7, 9],
                 'model__learning_rate': [0.01, 0.05, 0.1, 0.3],
                 'model__n_estimators': [100, 200, 300],
                 'model__min_child_weight': [1, 3, 5],
                 'model__subsample': [0.8, 0.9, 1.0],
                 'model__colsample_bytree': [0.8, 0.9, 1.0],
                 'model__gamma': [0, 0.1, 0.2],
                 'model__scale_pos_weight': [2.77]  
            }
            
            search = RandomizedSearchCV(
                model_pipeline,
                param_distributions=param_grid,
                n_iter=50,  
                scoring='f1', 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                random_state=42,
                n_jobs=-1,
                verbose=2
                )
            
            search.fit(x_train, y_train)
            model_pipeline = search.best_estimator_

            

            print("Best parameters:", search.best_params_)
            print("F1 score:", search.best_score_)

            y_pred = model_pipeline.predict(x_test)
            prob = model_pipeline.predict_proba(x_test)[:, 1]

            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, prob)

            print("Precision",precision)
            print("recall",recall)
            print("best_f1",f1)
            print("roc_auc_score",roc_auc)

            fold=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
            cross=cross_val_score(model_pipeline,x_train,y_train,cv=fold,scoring="roc_auc")

            print("cross_validation_score:",cross.mean())

            prob=model_pipeline.predict_proba(x_test)[:,1]
            roc_auc=roc_auc_score(y_test,prob)
            
            

            best_threshold=0.5
            best_f1=0

            for i in np.arange(0.1, 0.96, 0.05):
                y_prob = (prob > i).astype("int")
                f1_t = f1_score(y_test, y_prob)
                prec = precision_score(y_test, y_prob)
                rec = recall_score(y_test, y_prob)

                print(f"\nThreshold {i}")
                print("Precision:", prec)
                print("Recall:", rec)

                if f1_t>best_f1:
                    best_f1=f1_t
                    best_threshold=i
                    best_precision = prec  
                    best_recall = rec

            mlflow.log_param("model","XGBoost")

            for param_key,param_value in search.best_params_.items():
                mlflow.log_param(param_key,param_value)
            

            mlflow.log_metric("best_threshold",best_threshold)
            mlflow.log_metric("precision_at_0.5",precision)
            mlflow.log_metric("recall_at_0.5",recall)
            mlflow.log_metric("roc_auc_cv",cross.mean())
            mlflow.log_metric("roc_auc_score",roc_auc)
            mlflow.log_metric("f1_score",f1)
            mlflow.log_metric("precision_at_best_threshold", best_precision)
            mlflow.log_metric("recall_at_best_threshold", best_recall)
            mlflow.log_metric("best_f1",best_f1)
            
            os.makedirs("models", exist_ok=True)

            # save model
            joblib.dump(model_pipeline, "models/model.pkl")

            mlflow.sklearn.log_model(model_pipeline,"Customer Churn Prediction and Segmentation")

            print("Model saved successfully!")

            return model_pipeline
