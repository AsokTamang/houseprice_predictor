import logging
from src.utils import load_object
import numpy as np
from src.exception import CustomError
import os
import sys

class PredictPipeline:
    def __init__(self):
        self.prepipeline = load_object(os.path.join('artifacts','prepipeline.pkl'))  #loading the prepipeline object which is used for initial data transformation
        self.feature_selector = load_object(os.path.join('artifacts','feature_selection.pkl'))  #loading the feature selector object which is used for feature selection
        self.preprocessor = load_object(os.path.join('artifacts','preprocessor.pkl'))  #loading the preprocessor object which is used for data transformation
        self.model = load_object(os.path.join('artifacts','model.pkl'))  #loading the trained model which is used for prediction
    def predict(self,features):
        try:
            print("DEBUG: logging =", logging)
              
            features = self.prepipeline.transform(features)  #applying the initial data transformation on the input
            logging.info("Initial data transformation applied on the input features")  #logging the initial data transformation applied on the input features
            features = self.feature_selector.transform(features)  #applying feature selection on the input features
            logging.info("Feature selection applied on the input features")  #logging the feature selection applied on the input features
            features = self.preprocessor.transform(features)  #applying the data transformation on the input features
            logging.info("Data transformation applied on the input features")  #logging the data transformation applied on the input features   
            
            predicted_price =self.model.predict(features)  #predicting the target variable using the trained model on the input features
            logging.info(f"The predicted price of the house is {predicted_price[0]}")  #logging the predicted price of the house
            return np.expm1(predicted_price)   #as we have used log in the target variable during traininig, so converting the predicted price back into the oringinal price
        except Exception as e:
            import traceback
            traceback.print_exc()  # ← prints full stack trace to console
            raise CustomError(e, sys)