import os
import sys
from dataclasses import dataclass
from pathlib import Path
import joblib
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


BASE_DIR = Path(__file__).resolve().parents[3]


@dataclass
class ModelEvaluationConfig:
    evaluation_report_path: str = str(
        BASE_DIR / "artifacts" / "nlp" / "evaluation.json"
    )


class ModelEvaluator:

    def __init__(self):
        self.config = ModelEvaluationConfig()

    def evaluate_model(self, model, X_test, y_test):

        try:

            logging.info("Starting model evaluation")

            # prediction
            y_pred = model.predict(X_test)

            # metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            evaluation_metrics = {

                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

            logging.info(f"Evaluation Metrics: {evaluation_metrics}")

            # create directory
            os.makedirs(os.path.dirname(self.config.evaluation_report_path), exist_ok=True)

            # save report
            with open(self.config.evaluation_report_path, "w") as f:
                json.dump(evaluation_metrics, f, indent=4)

            logging.info(f"Evaluation report saved at {self.config.evaluation_report_path}")

            return evaluation_metrics

        except Exception as e:
            logging.error("Error occurred in model evaluation")
            raise CustomException(e, sys)