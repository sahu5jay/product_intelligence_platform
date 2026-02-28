


# src/structured_ml/components/data_ingestion.py

import os
import sys
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion process started")

        try:
            dataset_path = os.path.join("notebook/data/structured", "train.csv")
            df = pd.read_csv(dataset_path)
            logging.info(f"Dataset read successfully from {dataset_path}, shape: {df.shape}")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(
                f"Train and test data saved at {self.ingestion_config.train_data_path} and {self.ingestion_config.test_data_path}"
            )

            logging.info("Data Ingestion process completed successfully")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error("Error occurred in Data Ingestion")
            raise CustomException(e, sys)