# src/nlp_module/components/text_ingestion.py

import sys
import os
import pandas as pd
from pathlib import Path
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.config_loader import load_config
from src.shared_utils.constants import BASE_DIR

# ------------------------------
# Load config
# ------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "src/nlp_module/config.yaml"
config = load_config(CONFIG_PATH)

RAW_DATA_PATH = Path(config["text_ingestion"]["raw_data_path"])
DATASET_PATH = Path(config["text_ingestion"]["dataset_path"])


class TextIngestion:
    """
    Handles ingestion of raw text datasets:
    - Reads raw CSV dataset
    - Saves a copy to artifacts for reproducibility
    """

    def __init__(self, dataset_path: Path = DATASET_PATH, raw_save_path: Path = RAW_DATA_PATH):
        self.dataset_path = dataset_path
        self.raw_save_path = raw_save_path

        # Ensure the artifacts folder exists
        os.makedirs(self.raw_save_path.parent, exist_ok=True)

    def initiate_text_ingestion(self):
        """
        Read the raw CSV and save a copy to artifacts folder.
        Returns the path to the raw CSV.
        """
        try:
            logging.info(f"Reading raw dataset from {self.dataset_path}")
            df = pd.read_csv(self.dataset_path)
            logging.info(f"Dataset shape: {df.shape}")

            # Save a copy to artifacts/raw.csv
            df.to_csv(self.raw_save_path, index=False)
            logging.info(f"Raw dataset saved at {self.raw_save_path}")

            return str(self.raw_save_path)

        except Exception as e:
            logging.error("Error in text ingestion")
            raise CustomException(e, sys)