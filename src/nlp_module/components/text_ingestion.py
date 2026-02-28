import os
import sys
from dataclasses import dataclass
import pandas as pd

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


@dataclass
class TextIngestionConfig:
    raw_text_path: str = os.path.join("notebook","data","text","IMDB.csv")
    processed_text_path: str = os.path.join("artifacts","nlp", "processed_reviews.csv")


class TextIngestion:
    def __init__(self):
        self.ingestion_config = TextIngestionConfig()

    def initiate_text_ingestion(self):
        logging.info("Text Data Ingestion started")

        try:
            raw_path = self.ingestion_config.raw_text_path

            if not os.path.exists(raw_path):
                raise FileNotFoundError(f"Text dataset not found: {raw_path}")

            df = pd.read_csv(raw_path)
            logging.info(f"Dataset loaded with shape {df.shape}")

            # Basic cleaning
            df = df.dropna()
            df["review"] = df["review"].str.lower()

            os.makedirs(os.path.dirname(self.ingestion_config.processed_text_path), exist_ok=True)
            df.to_csv(self.ingestion_config.processed_text_path, index=False)

            logging.info("Processed text data saved successfully")

            return self.ingestion_config.processed_text_path

        except Exception as e:
            logging.error("Error in Text Data Ingestion")
            raise CustomException(e, sys)