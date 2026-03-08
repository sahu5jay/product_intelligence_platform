import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.config_loader import load_config

# Load config
BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "src/structured_ml/config.yaml"
config = load_config(CONFIG_PATH)

class DataIngestion:
    def __init__(self):
        self.dataset_path = BASE_DIR / config["data_ingestion"]["dataset_path"]
        self.raw_data_path = BASE_DIR / config["data_ingestion"]["raw_data_path"]
        self.train_data_path = BASE_DIR / config["data_ingestion"]["train_data_path"]
        self.test_data_path = BASE_DIR / config["data_ingestion"]["test_data_path"]
        self.test_size = config["data_ingestion"]["test_size"]
        self.random_state = config["data_ingestion"]["random_state"]

    def initiate_data_ingestion(self):
        """
        1. Load original dataset
        2. Save raw dataset
        3. Split into train/test
        4. Save train/test datasets
        5. Return paths
        """
        try:
            logging.info(f"Loading dataset from {self.dataset_path}")
            df = pd.read_csv(self.dataset_path)

            # Save raw dataset
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
            df.to_csv(self.raw_data_path, index=False)
            logging.info(f"Raw dataset saved at {self.raw_data_path}")

            # Split train/test
            train_df, test_df = train_test_split(
                df, test_size=self.test_size, random_state=self.random_state
            )

            # Save train/test
            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.test_data_path), exist_ok=True)

            train_df.to_csv(self.train_data_path, index=False)
            test_df.to_csv(self.test_data_path, index=False)
            logging.info(f"Train dataset saved at {self.train_data_path}")
            logging.info(f"Test dataset saved at {self.test_data_path}")

            return self.raw_data_path, self.train_data_path, self.test_data_path

        except Exception as e:
            logging.error("Error during data ingestion")
            raise CustomException(e, sys)