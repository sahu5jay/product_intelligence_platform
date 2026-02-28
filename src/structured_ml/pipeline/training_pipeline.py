# src/structured_ml/pipeline/training_pipeline.py

from src.structured_ml.components.data_ingestion import DataIngestion
from src.structured_ml.components.data_validation import DataValidation
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
import sys




if __name__ == "__main__":
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)

    logging.info("Data Ingestion Completed")

    # Step 2: Data Validation
    validation_obj = DataValidation()
    validation_report_path = validation_obj.initiate_data_validation()

    logging.info("Data Validation Completed")