import os
import sys
import json
import joblib
from dataclasses import dataclass
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
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
            logging.info("Model evaluation started")

            # Load trained model
            model = joblib.load(self.config.model_path)

            # Split features and target
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            class_report = classification_report(
                y_test,
                y_pred,
                output_dict=True
            )

            results = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "classification_report": class_report
            }

            # Save evaluation report
            os.makedirs(
                os.path.dirname(self.config.evaluation_report_path),
                exist_ok=True
            )

            with open(self.config.evaluation_report_path, "w") as f:
                json.dump(results, f, indent=4)

            logging.info("Model evaluation completed")
            logging.info(f"Report saved at {self.config.evaluation_report_path}")

            return results

        except Exception as e:
            logging.error("Error occurred in Model Evaluation")
            raise CustomException(e, sys)