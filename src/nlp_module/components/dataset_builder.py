# module/components/dataset_builder.py

import os
import sys
import logging
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.shared_utils.exception import CustomException


@dataclass
class DatasetBuilderConfig:
    train_file_path: str = os.path.join("artifacts", "dataset", "train.csv")
    test_file_path: str = os.path.join("artifacts", "dataset", "test.csv")


class DatasetBuilder:
    def __init__(self):
        self.config = DatasetBuilderConfig()

    def build_dataset(
        self,
        input_file_path: str,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        try:
            logging.info("Reading raw dataset")

            df = pd.read_csv(input_file_path)

            logging.info(f"Raw dataset shape: {df.shape}")

            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state
            )

            os.makedirs(os.path.dirname(self.config.train_file_path), exist_ok=True)

            train_df.to_csv(self.config.train_file_path, index=False)
            test_df.to_csv(self.config.test_file_path, index=False)

            logging.info("Train/Test split completed")
            logging.info(f"Train shape: {train_df.shape}")
            logging.info(f"Test shape: {test_df.shape}")

            return (
                self.config.train_file_path,
                self.config.test_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)