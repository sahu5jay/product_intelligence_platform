import os
import sys
import json
import joblib
from dataclasses import dataclass
from pathlib import Path
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


BASE_DIR = Path(__file__).resolve().parents[3]


@dataclass
class ModelEvaluationConfig:
    model_path: str = str(
        BASE_DIR / "artifacts" / "structured" / "model.pkl"
    )
    evaluation_report_path: str = str(
        BASE_DIR / "artifacts" / "structured" / "model_evaluation_report.json"
    )


class ModelEvaluation:

    def __init__(self):
        self.config = ModelEvaluationConfig()

    def initiate_model_evaluation(self, test_array):

        try:
            logging.info("Model evaluation started (Regression)")

            model = joblib.load(self.config.model_path)

            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            results = {
                "r2_score": r2,
                "mean_squared_error": mse,
                "mean_absolute_error": mae
            }

            os.makedirs(os.path.dirname(self.config.evaluation_report_path), exist_ok=True)

            with open(self.config.evaluation_report_path, "w") as f:
                json.dump(results, f, indent=4)

            logging.info("Model evaluation completed")
            logging.info(f"Report saved at {self.config.evaluation_report_path}")

            return results

        except Exception as e:
            logging.error("Error occurred in Model Evaluation")
            raise CustomException(e, sys)