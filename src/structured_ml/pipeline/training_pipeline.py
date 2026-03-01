# src/structured_ml/pipeline/training_pipeline.py

from src.structured_ml.components.data_ingestion import DataIngestion
from src.structured_ml.components.data_validation import DataValidation
from src.structured_ml.components.data_transformation import DataTransformation
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
import sys

if __name__ == "__main__":
    try:
        # -----------------------------
        # Step 1: Data Ingestion
        # -----------------------------
        obj = DataIngestion()
        raw_data_path, train_data_path, test_data_path = obj.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")

        # -----------------------------
        # Step 2: Data Validation
        # -----------------------------
        validation_obj = DataValidation()
        validation_report_path = validation_obj.initiate_data_validation()
        logging.info(f"Data Validation Completed. Report at: {validation_report_path}")

        # -----------------------------
        # Step 3: Data Transformation
        # -----------------------------
        transformation_obj = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation_obj.initiate_data_transformation(
            train_path=train_data_path,
            test_path=test_data_path,
            target_column="SalePrice"
        )

        logging.info("Data Transformation Completed")

    except Exception as e:
        logging.error("Exception in training pipeline")
        raise CustomException(e, sys)