import os
import sys
import dill
import numpy as np
from src.exception import CustomError
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import logging

def save_object(file_path, obj):    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # Create the directory if it doesn't exist
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)  # Use dill to serialize the object
            logging.info(f"Object saved successfully at {file_path}")  #logging the file path where the object is saved
    except Exception as e:
        raise CustomError(e, sys)

def load_object(file_path): #loading the trained objects 
    try:
        with open(file_path, 'rb') as file_obj:
            logging.info(f"Loading object from {file_path}")  #logging the file path from which the object is being loaded
            return dill.load(file_obj)  # Use dill to deserialize the object
    except Exception as e:
        raise CustomError(e, sys)    

def model_evaluation(X_train, y_train, X_test, models, params):
    try:
        score_report = {}
        trained_models = {}
        for model_name, model in models.items():
            param = params[model_name]
            gs = GridSearchCV(model, param, cv=5)   #cross_validation to find the best hyperparameters for the current model based on the given parameters for each model
            gs.fit(X_train, y_train)  #training the model on training data with different hyperparameters and finding the best hyperparameters based on cross validation score
            best_model = gs.best_estimator_  #finding the best model based on the best hyperparameters found by grid search for the current model
            logging.info(f"Best hyperparameters for {model_name} are {gs.best_params_}")  #logging the best hyperparameters found by grid search for the current model
            trained_models[model_name] = best_model  #finding the score of the best model on the cross_validation data
            cv_score = gs.best_score_
            score_report[model_name] = cv_score
           
        best_model_name = max(score_report, key=score_report.get)  #extracting the name of the best model based on the highest cross validation score among all the models
        best_model = trained_models[best_model_name]    #extracting the best model based on the name of the best model found by best score report
        test_predictions = best_model.predict(X_test)  #predicting the target variable on the test data using the best model found by grid search
        best_model_score = score_report[best_model_name]
        return best_model, best_model_score,np.expm1(test_predictions)
    except Exception as e:
        raise CustomError(e, sys)    