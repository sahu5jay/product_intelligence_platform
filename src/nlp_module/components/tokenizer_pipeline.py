# src/nlp_module/components/tokenizer_pipeline.py

import sys
import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.config_loader import load_config
from src.shared_utils.constants import BASE_DIR, NLP_TOKENIZER_DIR

# ------------------------------
# Load config
# ------------------------------
CONFIG_PATH = BASE_DIR / "src" / "nlp_module" / "config.yaml"
config = load_config(CONFIG_PATH)

TOKENIZER_CONFIG = config["tokenizer"]

# TOKENIZER_CONFIG = config["tokenizer"]
VOCAB_SIZE = TOKENIZER_CONFIG.get("vocab_size", 20000)
OOV_TOKEN = TOKENIZER_CONFIG.get("oov_token", "<OOV>")
MAX_LENGTH = TOKENIZER_CONFIG.get("max_length", 200)
PADDING_TYPE = TOKENIZER_CONFIG.get("padding_type", "post")
TRUNC_TYPE = TOKENIZER_CONFIG.get("trunc_type", "post")
# TOKENIZER_PATH = Path(TOKENIZER_CONFIG.get("tokenizer_path", NLP_ARTIFACTS / "tokenizer.pkl"))


class TokenizerPipeline:
    """
    Handles TF-IDF vectorization of text data and saves tokenizer for inference.
    """

    def __init__(self):
        # Ensure directory exists
        # os.makedirs(TOKENIZER_PATH.parent, exist_ok=True)
        self.vectorizer = TfidfVectorizer(max_features=VOCAB_SIZE)

    def initiate_tokenizer_transformation(self, train_text: pd.Series, test_text: pd.Series):
        """
        Fit TF-IDF vectorizer on training text and transform both train and test.
        Returns the transformed arrays and path to saved tokenizer.
        """
        try:
            logging.info("Fitting TF-IDF vectorizer on training text...")
            X_train_arr = self.vectorizer.fit_transform(train_text).toarray()
            X_test_arr = self.vectorizer.transform(test_text).toarray()

            # Save the vectorizer for future inference
            Path(NLP_TOKENIZER_DIR).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.vectorizer, NLP_TOKENIZER_DIR)
            logging.info(f"Tokenizer saved at: {NLP_TOKENIZER_DIR}")

            return X_train_arr, X_test_arr, str(NLP_TOKENIZER_DIR)

        except Exception as e:
            logging.error("Error in tokenizer pipeline")
            raise CustomException(e, sys)