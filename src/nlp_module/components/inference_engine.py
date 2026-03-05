# src/nlp_module/components/inference_engine.py

import sys
import joblib

from src.shared_utils.exception import CustomException
from src.shared_utils.logger import logging


class SentimentInferenceEngine:

    def __init__(self):

        try:
            self.model_path = "artifacts/nlp/sentiment_model.pkl"
            self.tokenizer_path = "artifacts/nlp/tokenizer.pkl"

            logging.info("Loading tokenizer")
            self.tokenizer = joblib.load(self.tokenizer_path)

            logging.info("Loading sentiment model")
            self.model = joblib.load(self.model_path)

            logging.info("Inference Engine initialized successfully")

        except Exception as e:
            logging.error("Error while loading model or tokenizer")
            raise CustomException(e, sys)

    def predict_sentiment(self, text):

        try:
            logging.info("Transforming input text")

            text_vector = self.tokenizer.transform([text])

            prediction = self.model.predict(text_vector)[0]

            return prediction

        except Exception as e:
            logging.error("Error during prediction")
            raise CustomException(e, sys)