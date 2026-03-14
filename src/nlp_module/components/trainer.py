import sys
from sklearn.linear_model import LogisticRegression

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
# from src.shared_utils import save_object
from src.shared_utils.utils import save_object, ensure_dir
from src.shared_utils.constants import TRAIN_DATA_PATH, TEST_DATA_PATH
from src.shared_utils.constants import NLP_MODEL_DIR, NLP_SENTIMENT_DIR


class Trainer:
    """
    Trainer class responsible for training and saving NLP model.
    """

    def __init__(self, config: dict):

        try:
            self.config = config

            # Model configuration
            self.model_type = config["model"]["model_type"]
            # self.model_path = config["model"]["model_path"]

            # Training configuration
            self.max_iter = config.get("training", {}).get("max_iter", 500)

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_training(self, X_train, y_train):

        try:

            logging.info("Initializing NLP model")

            # -------------------------
            # Model Selection
            # -------------------------
            if self.model_type == "logistic_regression":

                model = LogisticRegression(
                    max_iter=self.max_iter,
                    solver="liblinear"
                )

            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            # -------------------------
            # Training
            # -------------------------
            logging.info("Training model started")

            model.fit(X_train, y_train)

            logging.info("Model training completed")

            # -------------------------
            # Save Model
            # -------------------------
            # ensure_dir(str(NLP_ARTIFACTS / "model"))

            save_object(NLP_SENTIMENT_DIR, model)

            logging.info(f"Model saved at {NLP_SENTIMENT_DIR}")

            return model

        except Exception as e:

            logging.error("Error during model training")

            raise CustomException(e, sys)