import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.utils import save_object
from src.shared_utils.config_loader import load_config
from src.shared_utils.constants import PREPROCESSOR_PATH, STRUCTURED_MODEL_PATH


# Load configuration
config_path = "src/structured_ml/config.yaml"
config = load_config(config_path)
model_config = config["model_trainer"]


class ModelTrainer:
    """
    Train a regression model (Random Forest) using preprocessed data
    """
    def __init__(self):
        self.model_path = model_config["model_path"]
        self.n_estimators = model_config.get("n_estimators", 100)
        self.random_state = model_config.get("random_state", 42)

    def initiate_model_training(self, train_array: np.ndarray, test_array: np.ndarray):
        """
        Train the model, evaluate it, and return model + metrics
        """
        try:
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

            # Predict and evaluate
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            mse_test = mean_squared_error(y_test, y_test_pred)
            mae_test = mean_absolute_error(y_test, y_test_pred)

            logging.info(f"Train R2 Score: {r2_train:.4f}")
            logging.info(f"Test R2 Score: {r2_test:.4f}")
            logging.info(f"Test MSE: {mse_test:.4f}")
            logging.info(f"Test MAE: {mae_test:.4f}")

            # Save trained model
            save_object(self.model_path, model)
            logging.info(f"Trained model saved at {self.model_path}")

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

