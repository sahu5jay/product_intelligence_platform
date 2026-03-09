import sys
import joblib

from src.shared_utils.exception import CustomException
from src.shared_utils.logger import logging
from src.shared_utils.constants import NLP_MODEL_PATH, NLP_TOKENIZER_PATH


class PredictPipeline:

    def __init__(self):
        try:
            logging.info("Initializing NLP Prediction Pipeline")

            # Load tokenizer
            logging.info(f"Loading tokenizer from {NLP_TOKENIZER_PATH}")
            self.tokenizer = joblib.load(NLP_TOKENIZER_PATH)

            # Load model
            logging.info(f"Loading model from {NLP_MODEL_PATH}")
            self.model = joblib.load(NLP_MODEL_PATH)

            logging.info("Model and tokenizer loaded successfully")

        except Exception as e:
            logging.error("Error while loading model or tokenizer")
            raise CustomException(e, sys)

    def predict(self, text_input):

        try:
            logging.info("Starting sentiment prediction")

            # Ensure input is list
            if isinstance(text_input, str):
                text_input = [text_input]

            # Convert text to vector
            text_vector = self.tokenizer.transform(text_input)

            # Model prediction
            prediction = self.model.predict(text_vector)

            result = prediction[0]

            # Convert numeric label to sentiment
            label_map = {
                0: "Negative",
                1: "Positive"
            }

            sentiment = label_map.get(result, result)

            logging.info(f"Predicted sentiment: {sentiment}")

            return sentiment

        except Exception as e:
            logging.error("Error during prediction")
            raise CustomException(e, sys)