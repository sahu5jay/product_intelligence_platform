# src/structured_ml/pipeline/training_pipeline.py

import sys
from pathlib import Path
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.utils import save_object

from src.structured_ml.components.data_ingestion import DataIngestion
from src.structured_ml.components.data_validation import DataValidation
from src.structured_ml.components.data_transformation import DataTransformation
from src.structured_ml.components.model_trainer import ModelTrainer
from src.structured_ml.components.model_evaluation import ModelEvaluation
from src.structured_ml.components.model_saver import ModelSaver

from src.shared_utils.constants import STRUCTURED_ARTIFACTS, STRUCTURED_MODEL_PATH

# Fixed paths
EVAL_PATH = STRUCTURED_ARTIFACTS / "report/model_evaluation_report.json"

if __name__ == "__main__":

    try:
        logging.info("===== Structured ML Training Pipeline Started =====")

        # -----------------------------
        # Step : Data Ingestion
        # -----------------------------
        logging.info("Step 1: Data Ingestion Started")
        data_ingestion_obj = DataIngestion()
        raw_path, train_path, test_path = data_ingestion_obj.initiate_data_ingestion()
        logging.info(f"Data Ingestion Completed: Raw={raw_path}, Train={train_path}, Test={test_path}")

        # -----------------------------
        # Step : Data Validation
        # -----------------------------
        logging.info("Step 2: Data Validation Started")
        data_validation_obj = DataValidation(raw_path)
        validation_report = data_validation_obj.validate_data()
        logging.info(f"Data Validation Completed. Report: {validation_report}")

        if validation_report["status"] == "Invalid":
            logging.warning("Data Validation Failed: Missing Columns Detected. Check validation report before proceeding.")

        # -----------------------------
        # Step : Data Transformation
        # -----------------------------
        logging.info("Step 3: Data Transformation Started")
        data_transformation_obj = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation_obj.initiate_data_transformation(
            train_path, test_path
        )
        logging.info("Data Transformation Completed.")
        logging.info(f"Preprocessor saved at {preprocessor_path}")

        # -----------------------------
        # Step : Model Training
        # -----------------------------
        logging.info("Step 4: Model Training Started")
        model_trainer_obj = ModelTrainer()
        trainer_output = model_trainer_obj.initiate_model_training(train_arr, test_arr)

        # Correct keys
        r2_train = trainer_output["r2_train"]
        r2_test = trainer_output["r2_test"]
        mse_test = trainer_output["mse_test"]
        mae_test = trainer_output["mae_test"]
        model = trainer_output["model"]

        logging.info(f"Model Training Completed. Train R2 = {r2_train:.4f}, Test R2 = {r2_test:.4f}")
        logging.info(f"Test MSE = {mse_test:.4f}, Test MAE = {mae_test:.4f}")

        # # Save model
        # save_object(str(STRUCTURED_MODEL_PATH), model)
        # logging.info(f"Trained model saved at {STRUCTURED_MODEL_PATH}")

        # -----------------------------
        # Step 5: Model Evaluation
        # -----------------------------
        logging.info("Step 5: Model Evaluation Started")
        model_eval_obj = ModelEvaluation(
            model_path=str(STRUCTURED_MODEL_PATH),
            evaluation_report_path=str(EVAL_PATH)
        )
        evaluation_report = model_eval_obj.initiate_model_evaluation(test_arr)
        logging.info(f"Model Evaluation Completed. Report saved at {EVAL_PATH}")
        print(" Model Evaluation Metrics:", evaluation_report)

        # logging.info("===== Structured ML Training Pipeline Completed Successfully =====")

        saver = ModelSaver()
        final_model_path = saver.save_model(model)
        logging.info(f"Trained model saved at {final_model_path}")

    except Exception as e:
        logging.error("Error in training pipeline")
        raise CustomException(e, sys)