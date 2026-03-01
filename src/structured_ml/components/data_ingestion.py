import os
import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


BASE_DIR = Path(__file__).resolve().parents[3]


@dataclass
class DataIngestionConfig:
    raw_data_path: str = str(BASE_DIR / "artifacts" / "structured" / "raw.csv")
    train_data_path: str = str(BASE_DIR / "artifacts" / "structured" / "train.csv")
    test_data_path: str = str(BASE_DIR / "artifacts" / "structured" / "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")

        try:
            dataset_path = BASE_DIR / "notebook" / "data" / "structured" / "train.csv"

            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found at {dataset_path}")

            df = pd.read_csv(dataset_path)
            logging.info(f"Dataset loaded successfully with shape {df.shape}")

            selected_cols = [
                'GrLivArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GarageCars',
                'Neighborhood', 'ExterQual', 'KitchenQual', 'SalePrice'
            ]
            df = df[[col for col in selected_cols if col in df.columns]]

            # Ensure artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save filtered raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Filtered raw dataset saved at {self.ingestion_config.raw_data_path} with shape {df.shape}")

            # Split
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data Ingestion completed successfully")

            return (
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error occurred during Data Ingestion")
            raise CustomException(e, sys)