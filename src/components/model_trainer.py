from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


train_data_path, test_data_path = DataIngestion().initiate_data_ingestion()  #initiating the data ingestion and getting the train and test data path    