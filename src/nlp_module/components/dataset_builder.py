# src/nlp_module/components/dataset_builder.py

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.config_loader import load_config
from src.shared_utils.constants import NLP_ARTIFACTS, BASE_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH

# ------------------------------
# Load config from src/nlp_module/config.yaml
# ------------------------------
CONFIG_PATH = BASE_DIR / "src" / "nlp_module" / "config.yaml"
config = load_config(CONFIG_PATH)

DATASET_CONFIG = config["dataset_builder"]
TEST_SIZE = DATASET_CONFIG.get("test_size", 0.2)
RANDOM_STATE = DATASET_CONFIG.get("random_state", 42)


class DatasetBuilder:
    """
    Builds train and test datasets from cleaned CSV and saves them.
    """

    def __init__(self, processed_csv_path: Path):
        self.processed_csv_path = Path(processed_csv_path)
        # Ensure the parent folder for train/test exists
        # os.makedirs(TRAIN_DATA_PATH.parent, exist_ok=True)

    def build_dataset(self):
        try:
            logging.info(f"Reading cleaned dataset from {self.processed_csv_path}")
            df = pd.read_csv(self.processed_csv_path)

            # Ensure required columns exist
            if "review" not in df.columns or "sentiment" not in df.columns:
                raise ValueError("Expected 'review' and 'sentiment' columns in cleaned CSV")

            logging.info("Splitting dataset into train and test sets")
            train_df, test_df = train_test_split(
                df,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=df["sentiment"]  # maintain label distribution
            )

            # Save datasets in artifacts/nlp/data/
            train_df.to_csv(TRAIN_DATA_PATH, index=False)
            test_df.to_csv(TEST_DATA_PATH, index=False)

            logging.info(f"Train dataset saved at {TRAIN_DATA_PATH} (shape: {train_df.shape})")
            logging.info(f"Test dataset saved at {TEST_DATA_PATH} (shape: {test_df.shape})")

            return str(TRAIN_DATA_PATH), str(TEST_DATA_PATH)

        except Exception as e:
            logging.error("Error in building dataset")
            raise CustomException(e, sys)