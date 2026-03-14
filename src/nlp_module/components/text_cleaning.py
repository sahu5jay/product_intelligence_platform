# src/nlp_module/components/text_cleaning.py

import sys
import os
import pandas as pd
import string
from pathlib import Path
from nltk.corpus import stopwords
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.config_loader import load_config
# from src.shared_utils.constants import

# Ensure NLTK stopwords are downloaded
import nltk
nltk.download('stopwords')
STOPWORDS = set(stopwords.words("english"))

class TextCleaning:
    """
    Cleans raw text data based on config settings.
    The cleaned text will overwrite the 'review' column in the processed CSV.
    """

    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by lowercasing, removing punctuation, and stopwords."""
        try:
            text = text.lower()
            text = text.translate(str.maketrans("", "", string.punctuation))
            text = " ".join([word for word in text.split() if word not in STOPWORDS])
            return text
        except Exception as e:
            logging.error(f"Error cleaning text: {text}")
            raise e

    def initiate_text_cleaning(self):
        """
        Reads raw CSV, cleans the text column, and saves processed CSV.
        Only the text column is cleaned; no extra columns are added.
        """
        try:
            logging.info(f"Reading raw text data from {self.raw_data_path}")
            df = pd.read_csv(self.raw_data_path)
            if "review" not in df.columns:
                raise ValueError("Expected a 'review' column in the raw CSV")

            logging.info("Cleaning text data...")
            df["review"] = df["review"].apply(self.clean_text)

            os.makedirs(self.processed_data_path.parent, exist_ok=True)

            # Save cleaned CSV
            df.to_csv(self.processed_data_path, index=False)
            logging.info(f"Processed text saved at {self.processed_data_path}")

            return str(self.processed_data_path)

        except Exception as e:
            logging.error("Error during text cleaning")
            raise CustomException(e, sys)