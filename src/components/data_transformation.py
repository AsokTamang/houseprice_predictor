import sys
import os
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from src.exception import CustomError
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts','preprocessor.pkl')  #location where the preprocessor object will be stored after transformation


#class for changing the data type of the features based on the domain knowledge
class TypeCaster(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.int_to_str_features:list[str] = ['MSSubClass', 'OverallCond']

    def fit(self, X:pd.DataFrame, y=None):
        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted yet.")
        X = X.copy()
        for feature in X.columns:
            if feature in self.int_to_str_features:
                X[feature] = X[feature].astype(str)
        logging.info(   
            f"TypeCaster: Transformed {self.int_to_str_features} features to string as they represent category rather than numerical imporatance. ")        
        return X        
       

class DomainImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.none_features =['poolqc', 'miscfeature', 'alley', 'fence', 'masvnrtype',
                     'fireplacequ', 'garagetype', 'garagefinish', 'garagequal',
                     'garagecond', 'bsmtexposure', 'bsmtfintype2', 'bsmtqual', 
                     'bsmtcond', 'bsmtfintype1']
        self.zero_features = ['masvnrarea', 'garageyrblt']
        self.lot_frontage_median:dict[str,float] = {}  #store the median values of 'LotFrontage' for each 'Neighborhood'
        self.electrical_mode:str = ''  #store the mode value of 'Electrical' column 
        self.global_lot_frontage_median:float = 0.0  #store the global median value of 'LotFrontage' in case there are neighborhoods in test data which are not present in training data
    #this function is for learning the parameters quired for imputation from the training data and storing those parameters in the instance variables of the class which will be used later for imputation in both training and test data 
    def fit(self, X:pd.DataFrame, y=None):
        if 'LotFrontage'  in X.columns and 'neighborhood' in X.columns:
                for neighborhood in X['neighborhood'].unique():
                    median_value = X.loc[X['neighborhood'] == neighborhood, 'lotFrontage'].median() #extracting the median value of 'LotFrontage' for each 'Neighborhood'
                    logging.info(f"DomainImputer: Calculated median value of 'LotFrontage' for neighborhood '{neighborhood}' is {median_value}.")
                    self.lot_frontage_median[neighborhood] = median_value  #storing the median values in a dictionary with neighborhood as key and median value as value
                    self.global_lot_frontage_median = np.mean(list(self.lot_frontage_median.values()))  #calculating the global median value of 'LotFrontage' by taking the mean of all the neighborhood median values
        if 'electrical' in X.columns:
            logging.info(f"DomainImputer: Calculated mode value of 'Electrical' column is {X['electrical'].mode()[0]}.")
            self.electrical_mode = X['electrical'].mode()[0]  #extracting the mode value of 'Electrical' column and storing it in a variable
        return self
    

    def transform(self, X):
        X = X.copy()
        for feature in X.columns:
            if feature in self.none_features:
                logging.info(f"DomainImputer: Filling missing values of '{feature}' with 'None' as it represents absence of the feature.")  
                X[feature] = X[feature].fillna('None')  #filling the missing values with 'None' for the features in none_features list
            elif feature in self.zero_features:
                logging.info(f"DomainImputer: Filling missing values of '{feature}' with 0 as it represents absence of the feature.")
                X[feature] = X[feature].fillna(0)  #filling the missing values with 0 for the features in zero_features list
            elif feature == 'lotFrontage':
                logging.info(f"DomainImputer: Filling missing values of 'LotFrontage' based on the median value for each neighborhood.")
                for neighborhood, median_value in self.lot_frontage_median.items():
                    if neighborhood in X['neighborhood'].unique():
                     X.loc[X['neighborhood'] == neighborhood, 'lotFrontage'] = X.loc[X['neighborhood'] == neighborhood, 'lotFrontage'].fillna(median_value)  #filling the missing values of 'LotFrontage' with the corresponding median value based on the neighborhood
                    else:
                        logging.info(f"DomainImputer: Neighborhood '{neighborhood}' not found in the data. Filling missing values of 'LotFrontage' with global median value {self.global_lot_frontage_median}.")
                        X['lotFrontage'] = X['lotFrontage'].fillna(self.global_lot_frontage_median)  #filling the missing values of 'LotFrontage' with global median value in case there are neighborhoods in test data which are not present in training data 
            elif feature == 'electrical':
                logging.info(f"DomainImputer: Filling missing values of 'Electrical' column with mode value '{self.electrical_mode}' as it is the most common value in the column.")
                X[feature] = X[feature].fillna(self.electrical_mode)  #filling the missing values of 'Electrical' column with the mode value
            return X    
        



class DropConstantNumerical(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.variance_threshold:float = 0.01  #threshold for variance below which the feature will be dropped

    def fit(self, X:pd.DataFrame, y=None):
        X = X.copy()
        target = 'saleprice'
        numerical = [feature for feature in X.select_dtypes(include=np.number).columns if feature!=target]
        categorical = [feature for feature in X.select_dtypes(exclude=np.number).columns if feature!=target]
        vt = VarianceThreshold(0.01)
        vt.fit(df[numerical])  #checking the constant features of numerical features based on threshold 0.01
        variance_result = vt.get_support()  #getting the result of checking the passed data with the threshold
        constant_num_features = [col for col,s in zip(df[numerical],variance_result) if not s]  #storing those numerical features whose variance is below 0.01
        df = df.drop(columns = constant_num_features)
        return df

    def transform(self, X):
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted yet.")
        X = X.copy()
        #dropping the 'id' column as it is not useful for prediction and it is just an identifier for each row
        if 'id' in X.columns:
            logging.info("DataCleaning: Dropping 'id' column as it is not useful for prediction.")
            X = X.drop('id', axis=1)