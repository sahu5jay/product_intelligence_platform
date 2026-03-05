import os
import sys
import joblib

from src.shared_utils.exception import CustomException
from src.shared_utils.logger import logging


class PredictPipeline:

    def __init__(self):

        try:
            logging.info("Initializing NLP Prediction Pipeline")

            self.model_path = os.path.join(
                "artifacts", "nlp", "sentiment_model.pkl"
            )

            self.tokenizer_path = os.path.join(
                "artifacts", "nlp", "tokenizer.pkl"
            )

            logging.info("Loading tokenizer...")
            self.tokenizer = joblib.load(self.tokenizer_path)

            logging.info("Loading sentiment model...")
            self.model = joblib.load(self.model_path)

            logging.info("Model and tokenizer loaded successfully")

        except Exception as e:
            logging.error("Error while loading model or tokenizer")
            raise CustomException(e, sys)

    def predict(self, text_input):

        try:
            logging.info("Starting sentiment prediction")

            if isinstance(text_input, str):
                text_input = [text_input]

            text_vector = self.tokenizer.transform(text_input)

            prediction = self.model.predict(text_vector)

            logging.info(f"Prediction: {prediction}")

            return prediction

        except Exception as e:
            logging.error("Error during prediction")
            raise CustomException(e, sys)