import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.utils import save_object
from src.shared_utils.config_loader import load_config
from src.shared_utils.constants import STRUCTURED_MODEL_PATH


# Load configuration
config_path = "src/structured_ml/config.yaml"
config = load_config(config_path)
model_config = config["model_trainer"]


class ModelTrainer:
    """
    Train a regression model (Random Forest)
    """

    def __init__(self, train_dataset, test_dataset):

        # Save dataset paths
        self.train_dataset = Path(train_dataset)
        self.test_dataset = Path(test_dataset)

        self.model_path = STRUCTURED_MODEL_PATH
        self.n_estimators = model_config.get("n_estimators", 100)
        self.random_state = model_config.get("random_state", 42)

    def initiate_model_training(self):

        try:

            logging.info("Loading train and test datasets")

            train_array = pd.read_csv(self.train_dataset).values
            test_array = pd.read_csv(self.test_dataset).values

            logging.info("Splitting features and target")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Initialize model
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state
            )

            logging.info("Training RandomForestRegressor")

            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Metrics
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            mse_test = mean_squared_error(y_test, y_test_pred)
            mae_test = mean_absolute_error(y_test, y_test_pred)

            logging.info(f"Train R2 Score: {r2_train:.4f}")
            logging.info(f"Test R2 Score: {r2_test:.4f}")

            # Save model
            os.makedirs(Path(self.model_path).parent, exist_ok=True)
            save_object(self.model_path, model)

            logging.info(f"Model saved at {self.model_path}")

            return {
                "model": model,
                "r2_train": r2_train,
                "r2_test": r2_test,
                "mse_test": mse_test,
                "mae_test": mae_test
            }

        except Exception as e:
            logging.error("Error during model training")
            raise CustomException(e, sys)