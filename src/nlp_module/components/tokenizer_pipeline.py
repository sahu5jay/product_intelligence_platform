# src/nlp_module/components/tokenizer_pipeline.py

import sys
import os
import joblib
import pandas as pd
from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.config_loader import load_config
from src.shared_utils.constants import NLP_ARTIFACTS

# Load config
CONFIG_PATH = NLP_ARTIFACTS.parent / "src" / "nlp_module" / "config.yaml"
config = load_config(CONFIG_PATH)
TOKENIZER_CONFIG = config["tokenizer"]
VOCAB_SIZE = TOKENIZER_CONFIG.get("vocab_size", 20000)
OOV_TOKEN = TOKENIZER_CONFIG.get("oov_token", "<OOV>")
MAX_LENGTH = TOKENIZER_CONFIG.get("max_length", 200)
PADDING_TYPE = TOKENIZER_CONFIG.get("padding_type", "post")
TRUNC_TYPE = TOKENIZER_CONFIG.get("trunc_type", "post")
TOKENIZER_PATH = Path(TOKENIZER_CONFIG.get("tokenizer_path", NLP_ARTIFACTS / "tokenizer.pkl"))


class TokenizerPipeline:
    """
    Tokenizes and pads text data for NLP model training.
    Saves tokenizer to artifacts for reuse.
    """

    def __init__(self, tokenizer_path: Path = TOKENIZER_PATH):
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None

    def fit_tokenizer(self, texts):
        try:
            self.tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
            self.tokenizer.fit_on_texts(texts)
            os.makedirs(self.tokenizer_path.parent, exist_ok=True)
            joblib.dump(self.tokenizer, self.tokenizer_path)
            logging.info(f"Tokenizer saved at {self.tokenizer_path}")
        except Exception as e:
            logging.error("Error fitting tokenizer")
            raise CustomException(e, sys)

    def transform_texts(self, texts):
        try:
            if self.tokenizer is None:
                if not self.tokenizer_path.exists():
                    raise Exception("Tokenizer not found. Please fit it first.")
                self.tokenizer = joblib.load(self.tokenizer_path)
                logging.info("Tokenizer loaded successfully")

            sequences = self.tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
            return padded
        except Exception as e:
            logging.error("Error transforming texts")
            raise CustomException(e, sys)

    def initiate_tokenizer_transformation(self, train_text, test_text):
        """
        Fit tokenizer on training text and transform train/test text.
        Returns padded arrays and tokenizer path.
        """
        try:
            self.fit_tokenizer(train_text)
            X_train_arr = self.transform_texts(train_text)
            X_test_arr = self.transform_texts(test_text)
            return X_train_arr, X_test_arr, str(self.tokenizer_path)
        except Exception as e:
            logging.error("Error in tokenizer pipeline")
            raise CustomException(e, sys)