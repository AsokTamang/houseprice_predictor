import sys
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from src.exception import CustomError
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts','preprocessor.pkl')  #location where the preprocessor object will be stored after transformation




