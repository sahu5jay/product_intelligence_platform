# src/nlp_module/components/evaluator.py

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.config_loader import load_config

BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "src" / "nlp_module" / "config.yaml"
config = load_config(CONFIG_PATH)


@dataclass
class ModelEvaluationConfig:
    metrics: list = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1_score"])
    evaluation_report_path: str = "artifacts/nlp/report/evaluation.json"


class ModelEvaluator:

    def __init__(self):
        self.config = ModelEvaluationConfig()

    def evaluate_model(self, model, X_test, y_test):

        try:
            logging.info("Starting model evaluation")

            # prediction
            y_pred = model.predict(X_test)

            evaluation_metrics = {}

            if "accuracy" in self.config.metrics:
                evaluation_metrics["accuracy"] = accuracy_score(y_test, y_pred)
            if "precision" in self.config.metrics:
                evaluation_metrics["precision"] = precision_score(y_test, y_pred, average='weighted')
            if "recall" in self.config.metrics:
                evaluation_metrics["recall"] = recall_score(y_test, y_pred, average='weighted')
            if "f1_score" in self.config.metrics:
                evaluation_metrics["f1_score"] = f1_score(y_test, y_pred, average='weighted')

            logging.info(f"Evaluation Metrics: {evaluation_metrics}")

            # create directory if not exists
            os.makedirs(os.path.dirname(self.config.evaluation_report_path), exist_ok=True)

            # save report
            with open(self.config.evaluation_report_path, "w") as f:
                json.dump(evaluation_metrics, f, indent=4)

            logging.info(f"Evaluation report saved at {self.config.evaluation_report_path}")

            return evaluation_metrics, self.config.evaluation_report_path

        except Exception as e:
            logging.error("Error occurred in model evaluation")
            raise CustomException(e, sys)