# src/nlp_module/components/model_loader.py

import sys
import os
import joblib
from pathlib import Path
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.config_loader import load_config
from src.shared_utils.constants import NLP_ARTIFACTS

# -------------------------
# Load config
# -------------------------
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
config = load_config(CONFIG_PATH)

MODEL_PATH = NLP_ARTIFACTS / "model" / "sentiment_model.pkl"
TOKENIZER_PATH = NLP_ARTIFACTS / "tokenizer" / "tokenizer.pkl"


class ModelLoader:
    """
    Loads the trained NLP model and tokenizer for inference.
    """

    def __init__(self, model_path: Path = MODEL_PATH, tokenizer_path: Path = TOKENIZER_PATH):
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)

    def load_model(self):
        """
        Load the trained model from disk.
        """
        try:
            logging.info(f"Loading trained model from {self.model_path}")
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            model = joblib.load(self.model_path)
            logging.info("Model loaded successfully")
            return model
        except Exception as e:
            logging.error("Error loading model")
            raise CustomException(e, sys)

    def load_tokenizer(self):
        """
        Load the trained tokenizer from disk.
        """
        try:
            logging.info(f"Loading tokenizer from {self.tokenizer_path}")
            if not self.tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer file not found at {self.tokenizer_path}")
            tokenizer = joblib.load(self.tokenizer_path)
            logging.info("Tokenizer loaded successfully")
            return tokenizer
        except Exception as e:
            logging.error("Error loading tokenizer")
            raise CustomException(e, sys)