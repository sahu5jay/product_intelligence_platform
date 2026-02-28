# src/structured_ml/pipeline/training_pipeline.py

from src.structured_ml.components.data_ingestion import DataIngestion

# from src.structured_ml.components.data_validation import DataValidation
# from src.structured_ml.components.data_transformation import DataTransformation
# from src.structured_ml.components.model_trainer import ModelTrainer
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
import sys


# class TrainingPipeline:
#     def __init__(self):
#         pass

    # def run_pipeline(self):
    #     logging.info("Training Pipeline started")
    #     # try:
    #         # Step 1: Data Ingestion
    #         data_ingestion = DataIngestion()
    #         train_path, test_path = data_ingestion.initiate_data_ingestion()

            # Step 2: Data Validation
            # data_validation = DataValidation()
            # validated_train, validated_test = data_validation.validate_data(train_path, test_path)

            # Step 3: Data Transformation
            # data_transformation = DataTransformation()
            # X_train, y_train, X_test, y_test = data_transformation.transform_data(validated_train, validated_test)

            # Step 4: Model Training
            # model_trainer = ModelTrainer()
            # model_trainer.train_model(X_train, y_train, X_test, y_test)

        #     logging.info("Training Pipeline completed successfully")

        # except Exception as e:
            # logging.error("Error occurred in Training Pipeline")
            # raise CustomException(e, sys)


if __name__ == "__main__":
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)