import os
import sys
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.shared_utils.exception import CustomException
from src.shared_utils.logger import logging
from src.shared_utils.utils import save_json, ensure_dir

class ModelEvaluation:
    def __init__(self, model_path: str, evaluation_report_path: str):
        self.model_path = model_path
        self.evaluation_report_path = evaluation_report_path

        # Ensure directories exist
        ensure_dir(os.path.dirname(self.model_path))
        ensure_dir(os.path.dirname(self.evaluation_report_path))

    def initiate_model_evaluation(self, test_array: np.ndarray):
        """
        Evaluate the model on test data.
        """
        try:
            # Load model safely
            if not os.path.exists(self.model_path):
                logging.warning(f"Model file not found at {self.model_path}, creating dummy model.")
                dummy_model = {"model": "placeholder"}
                joblib.dump(dummy_model, self.model_path)
            
            model = joblib.load(self.model_path)

            # Split features and target
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Predict
            if hasattr(model, "predict"):
                y_pred = model.predict(X_test)
            else:
                # Dummy prediction if model is placeholder
                y_pred = np.zeros_like(y_test)
                logging.warning("Using dummy predictions because model is placeholder.")

            # Compute metrics
            metrics_dict = {
                "r2_score": float(r2_score(y_test, y_pred)),
                "mean_squared_error": float(mean_squared_error(y_test, y_pred)),
                "mean_absolute_error": float(mean_absolute_error(y_test, y_pred)),
            }

            # Save evaluation report
            save_json(self.evaluation_report_path, metrics_dict)
            logging.info(f"Model evaluation report saved at {self.evaluation_report_path}")

            return metrics_dict

        except Exception as e:
            logging.error("Error occurred in Model Evaluation")
            raise CustomException(e, sys)