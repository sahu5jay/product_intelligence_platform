import os
import sys
import joblib
from dataclasses import dataclass
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


BASE_DIR = Path(__file__).resolve().parents[3]


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = str(
        BASE_DIR / "artifacts" / "structured" / "model.pkl"
    )


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        """
        train_array: numpy array (features + target)
        test_array: numpy array (features + target)
        """
        try:
            logging.info("Model training started")

            # Split features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Initialize model
            model = RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )

            # Train model
            model.fit(X_train, y_train)

            logging.info("Model training completed")

            # Evaluate on test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            logging.info(f"Model Accuracy: {accuracy}")

            # Save model
            os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
            joblib.dump(model, self.config.trained_model_path)

            logging.info(f"Model saved at {self.config.trained_model_path}")

            return {
                "accuracy": accuracy,
                "model_path": self.config.trained_model_path
            }

        except Exception as e:
            logging.error("Error occurred in Model Trainer")
            raise CustomException(e, sys)