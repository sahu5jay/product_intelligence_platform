import os
import sys
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException

# Import your cleaning function
from src.nlp_module.components.text_cleaning import clean_text

# Set base directory
BASE_DIR = Path(__file__).resolve().parents[3]


@dataclass
class TextIngestionConfig:
    raw_data_path: str = str(BASE_DIR / "notebook" / "data" / "text" / "IMDB.csv")
    artifact_raw_path: str = str(BASE_DIR / "artifacts" / "nlp" / "raw.csv")
    train_data_path: str = str(BASE_DIR / "artifacts" / "nlp" / "train.csv")
    test_data_path: str = str(BASE_DIR / "artifacts" / "nlp" / "test.csv")


class TextIngestion:
    def __init__(self):
        self.ingestion_config = TextIngestionConfig()

    def initiate_text_ingestion(self):
        logging.info("Text Data Ingestion started")

        try:
            raw_path = self.ingestion_config.raw_data_path

            if not os.path.exists(raw_path):
                raise FileNotFoundError(f"Dataset not found at {raw_path}")

            # Load dataset
            df = pd.read_csv(raw_path)
            logging.info(f"Dataset loaded successfully with shape: {df.shape}")

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw copy into artifacts
            df.to_csv(self.ingestion_config.artifact_raw_path, index=False)
            logging.info("Raw dataset saved into artifacts")

            # Validate required columns
            if "review" not in df.columns or "sentiment" not in df.columns:
                raise ValueError("Dataset must contain 'review' and 'sentiment' columns")

            # Drop nulls
            df.dropna(subset=["review", "sentiment"], inplace=True)

            # Clean text column
            logging.info("Cleaning text data...")
            # Convert all values to string first, then clean
            df["review"] = df["review"].astype(str).apply(clean_text)
            logging.info("Text cleaning completed")

            # Train-Test Split
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["sentiment"]
            )

            # Save train & test
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Train and Test files saved successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error occurred in Text Data Ingestion")
            raise CustomException(e, sys)