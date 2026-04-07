from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformer
from src.logger import logging
import pandas as pd


train_data_path, test_data_path = DataIngestion().initiate_data_ingestion()  #initiating the data ingestion and getting the train and test data path    
train_data = pd.read_csv(train_data_path)  #reading the train data
test_data = pd.read_csv(test_data_path)  #reading the test data
dt = DataTransformer()
train_data.columns = train_data.columns.str.lower()
test_data.columns = test_data.columns.str.lower()
X_train = train_data.drop(columns=['saleprice'])
y_train = train_data['saleprice']
dt.fit(X_train, y_train)  #fitting the data transformer on the training data to learn the parameters required for transformation and feature selection
X_train_encoded, y_train_logged = dt.transform(X_train, y_train)
X_test_encoded = dt.transform(test_data) #as the target saleprice is not present in the test data, so need for dropping the unavailable feature
logging.info('Data transformation completed successfully for both training and test data')
print('first five tranformed training data: \n', X_train_encoded[:5])
print('first five transformed test data: \n', X_test_encoded[:5])


