import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.utils import save_json
from src.shared_utils.constants import NLP_METRICS_JSON


class Evaluator:
    """
    Evaluates NLP model performance and saves metrics
    """

    def __init__(self, config: dict):
        try:
            self.config = config

            # metrics defined in config.yaml
            self.metrics_list = config.get("evaluation", {}).get("metrics", [])

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate(self, model, X_test, y_test):

        try:
            logging.info("Starting model evaluation")

            # -------------------------
            # Predictions
            # -------------------------
            y_pred = model.predict(X_test)

            metrics_result = {}

            # -------------------------
            # Metrics Calculation
            # -------------------------
            if "accuracy" in self.metrics_list:
                metrics_result["accuracy"] = accuracy_score(y_test, y_pred)

            if "precision" in self.metrics_list:
                metrics_result["precision"] = precision_score(
                    y_test, y_pred, average="weighted"
                )

            if "recall" in self.metrics_list:
                metrics_result["recall"] = recall_score(
                    y_test, y_pred, average="weighted"
                )

            if "f1_score" in self.metrics_list:
                metrics_result["f1_score"] = f1_score(
                    y_test, y_pred, average="weighted"
                )

            logging.info(f"Evaluation Metrics: {metrics_result}")

            # -------------------------
            # Save Metrics JSON
            # -------------------------
            save_json(NLP_METRICS_JSON, metrics_result)

            logging.info(f"Metrics saved at {NLP_METRICS_JSON}")

            return metrics_result

        except Exception as e:
            logging.error("Error during model evaluation")
            raise CustomException(e, sys)