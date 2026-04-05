import os
import sys
from src.exception import CustomError
from src.logger import logging
from dataclasses import dataclass
import pandas as pd

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('root_data','train_data.csv')
    test_data_path:str = os.path.join('root_data','test_data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  

    def initiate_data_ingestion(self):
        try:
            logging.info('Data Ingestion method started')
            #reading the data from source
            df_train = pd.read_csv(os.path.join('notebook/data','train.csv'))
            df_test = pd.read_csv(os.path.join('notebook/data','test.csv'))
            logging.info('Training and test Data read successfully')

            #creating root data folder if not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)  #creating the root data folder if the root data folder doesnot exist  

            #storing the data in train and test path
            df_train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            df_test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Data ingestion completed')
            return self.ingestion_config.train_data_path,self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomError(e,sys)


if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()