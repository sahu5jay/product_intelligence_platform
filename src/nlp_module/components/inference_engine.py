# src/nlp_module/components/inference_engine.py

import sys
import joblib
import os
from src.shared_utils.exception import CustomException
from src.shared_utils.logger import logging
from src.shared_utils.config_loader import load_config
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "src" / "nlp_module" / "config.yaml"
config = load_config(CONFIG_PATH)


class SentimentInferenceEngine:
    def __init__(self, model_path=None, tokenizer_path=None):
        try:
            # Use config paths if not provided
            self.model_path = model_path or str(BASE_DIR / config["inference"]["model_path"])
            self.tokenizer_path = tokenizer_path or str(BASE_DIR / config["inference"]["tokenizer_path"])

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            if not os.path.exists(self.tokenizer_path):
                raise FileNotFoundError(f"Tokenizer file not found at {self.tokenizer_path}")

            logging.info(f"Loading tokenizer from {self.tokenizer_path}")
            self.tokenizer = joblib.load(self.tokenizer_path)

            logging.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)

            logging.info("Inference Engine initialized successfully")

        except Exception as e:
            logging.error("Error while initializing Inference Engine")
            raise CustomException(e, sys)

    def predict_sentiment(self, text):
        try:
            logging.info("Transforming input text for prediction")
            text_vector = self.tokenizer.transform([text])
            prediction = self.model.predict(text_vector)[0]
            logging.info(f"Prediction: {prediction}")
            return prediction

        except Exception as e:
            logging.error("Error during prediction")
            raise CustomException(e, sys)