# src/nlp_module/components/dataset_builder.py

import os
import sys
import logging
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.shared_utils.exception import CustomException
from src.shared_utils.config_loader import load_config

# -------------------------------
# Paths and Config
# -------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "src" / "nlp_module" / "config.yaml"
config = load_config(CONFIG_PATH)

# -------------------------------
# Dataset Builder Config
# -------------------------------
@dataclass
class DatasetBuilderConfig:
    train_file_path: str = str(BASE_DIR / config["dataset_builder"]["train_data_path"])
    test_file_path: str = str(BASE_DIR / config["dataset_builder"]["test_data_path"])
    test_size: float = config["dataset_builder"]["test_size"]
    random_state: int = config["dataset_builder"]["random_state"]

# -------------------------------
# Dataset Builder Class
# -------------------------------
class DatasetBuilder:

    def __init__(self):
        self.config = DatasetBuilderConfig()

    def build_dataset(self, input_file_path: str):
        """
        Reads raw CSV dataset, splits into train/test, and saves to artifacts.
        """
        try:
            logging.info("Reading raw dataset")

            df = pd.read_csv(input_file_path)
            logging.info(f"Raw dataset shape: {df.shape}")

            # Train/test split
            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )

            # Ensure directories exist
            os.makedirs(os.path.dirname(self.config.train_file_path), exist_ok=True)

            # Save CSVs
            train_df.to_csv(self.config.train_file_path, index=False)
            test_df.to_csv(self.config.test_file_path, index=False)

            logging.info("Train/Test split completed")
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            return self.config.train_file_path, self.config.test_file_path

        except Exception as e:
            raise CustomException(e, sys)