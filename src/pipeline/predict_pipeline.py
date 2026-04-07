from src.utils import load_object
import numpy as np
from src.exception import CustomError
import os
import sys

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = os.path.join('artifacts','model.pkl')  #loading the trained model which is used for prediction
            prepipeline = load_object(os.path.join('artifacts','prepipeline.pkl'))  #loading the prepipeline object which is used for initial data transformation
            feature_selector = load_object(os.path.join('artifacts','feature_selection.pkl'))  #loading the feature selector object which is used for feature selection
            preprocessor = load_object(os.path.join('artifacts','preprocessor.pkl'))  #loading the preprocessor object which is used for data transformation
            features = prepipeline.transform(features)  #applying the initial data transformation on the input
            features = feature_selector.transform(features)  #applying feature selection on the input features
            features = preprocessor.transform(features)  #applying the data transformation on the input features
            
            model = load_object(model_path)  #loading the trained model which is used for prediction
            predicted_price = model.predict(features)  #predicting the target variable using the trained model on the input features
            return np.expm1(predicted_price) 
        except Exception as e:
            raise CustomError(e, sys)