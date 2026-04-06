import os
import sys
import dill
from src.logger import logging
from src.exception import CustomException


def save_object(file_path, obj):    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # Create the directory if it doesn't exist
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)  # Use dill to serialize the object
    except Exception as e:
        raise CustomException(e, sys)