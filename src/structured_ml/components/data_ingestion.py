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
    def __init__(self, raw_data_path):
        logging.info(f"--------------======---->>><< {BASE_DIR}")
        self.raw_data_path = Path(raw_data_path)

    def initiate_data_ingestion(self):
        """
        1. Load original dataset
        2. Save raw dataset
        3. Split into train/test
        4. Save train/test datasets
        5. Return paths
        """
        try:
            logging.info(f"Loading dataset from {self.raw_data_path}")
            df = pd.read_csv(self.raw_data_path)

            # Save raw dataset
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
            df.to_csv(self.raw_data_path, index=False)
            logging.info(f"Raw dataset saved at {self.raw_data_path}")

            return self.raw_data_path

        except Exception as e:
            logging.error("Error during data ingestion")
            raise CustomException(e, sys)