import sys
import joblib

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.constants import NLP_MODEL_PATH, TOKENIZER_PATH


class InferenceEngine:
    """
    Loads trained model and tokenizer and performs sentiment prediction
    """

    def __init__(self):

        try:
            logging.info("Loading NLP inference artifacts")

            # Load trained model
            self.model = joblib.load(NLP_MODEL_PATH)

            # Load tokenizer
            self.tokenizer = joblib.load(TOKENIZER_PATH)

            logging.info("Model and tokenizer loaded successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, text: str):

        try:
            logging.info("Running inference")

            # Convert input text to list
            input_text = [text]

            # Transform text using tokenizer
            vector = self.tokenizer.transform(input_text)

            # Predict sentiment
            prediction = self.model.predict(vector)

            sentiment = prediction[0]

            logging.info(f"Predicted sentiment: {sentiment}")

            return sentiment

        except Exception as e:
            raise CustomException(e, sys)