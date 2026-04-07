import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformer
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomError
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    traned_model_path:str = os.path.join('artifacts','model.pkl')



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def iniate_model_trainer(self,train_data,test_data):
        try:
            logging.info('Model Trainer method started')
            X_train,y_train = train_data[:,:-1],train_data[:,-1]  #separating the features and target variable from the transformed training data
            X_test = test_data
            models = {
                    "Ridge Regression": Ridge(),
                    "Random Forest": RandomForestRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "XGBRegressor": XGBRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                }
                #parameters for hyperparameter tuning based on each model
            params = {
                    "Ridge Regression": {
                        "alpha": [0.1, 1.0, 10.0],
                        "random_state": [42,55,60],
                       
                    },
                    "Random Forest": {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 10, 20],
                        "random_state": [42,55,60],
                    },
                    "Gradient Boosting": {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.05,0.01,0.1],
                        "random_state": [42,55,60],
                    },
                    "XGBRegressor": {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "random_state": [42,55,60],
                    },
                    "Decision Tree": {
                        "random_state": [42,55,60]
                        ,"max_depth": [None, 10, 20]},
                }
            best_model,best_model_score = model_evaluation(X_train, y_train, X_test, models,params)
            if best_model_score < 0.6:
                raise CustomError("No best model found with score greater than 0.6", sys)
            logging.info(f"Best model found on both training and testing dataset with score of {best_model_score}")
            save_object(file_path=self.config.trained_model_file_path, obj=best_model)
            logging.info(f"Trained model saved at {self.config.trained_model_file_path}")
            return best_model , best_model_score


               
        except Exception as e:
            raise CustomError(e,sys)    



train_data_path, test_data_path = DataIngestion().initiate_data_ingestion()  #initiating the data ingestion and getting the train and test data path    
train_data = pd.read_csv(train_data_path)  #reading the train data
test_data = pd.read_csv(test_data_path)  #reading the test data
dt = DataTransformer()
train_data.columns = train_data.columns.str.lower()
test_data.columns = test_data.columns.str.lower()
X_train = train_data.drop(columns=['saleprice'])
y_train = train_data['saleprice']
dt.fit(X_train, y_train)  #fitting the data transformer on the training data to learn the parameters required for transformation and feature selection
train_data_transformed = dt.transform(X_train, y_train)
test_data_transformed= dt.transform(test_data) #as the target saleprice is not present in the test data, so need for dropping the unavailable feature



