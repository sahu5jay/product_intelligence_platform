# structured_ml/components/model_trainer.py

import os
import sys
import logging
import numpy as np

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from src.shared_utils.exception import CustomException
from src.shared_utils.save_object import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts", "structured", "model.pkl"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            # Last column is target
            X_train, y_train = (
                train_array[:, :-1],
                train_array[:, -1],
            )

            X_test, y_test = (
                test_array[:, :-1],
                test_array[:, -1],
            )

            logging.info("Training Linear Regression model")

            model = LinearRegression()
            model.fit(X_train, y_train)

            logging.info("Model training completed")

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            logging.info(f"Model evaluation completed")
            logging.info(f"R2 Score: {r2}")
            logging.info(f"MSE: {mse}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info(
                f"Model saved at {self.model_trainer_config.trained_model_file_path}"
            )

            return {
                "r2_score": r2,
                "mse": mse,
                "model_path": self.model_trainer_config.trained_model_file_path
            }

        except Exception as e:
            raise CustomException(e, sys)