import os
import sys
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


BASE_DIR = Path(__file__).resolve().parents[3]


@dataclass
class DataValidationConfig:
    raw_data_path: str = str(BASE_DIR / "artifacts" / "structured" / "raw.csv")
    validation_report_path: str = str(BASE_DIR / "artifacts" / "structured" / "validation_report.txt")



class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()

    def initiate_data_validation(self):
        logging.info("Structured Data Validation started")
        logging.info(BASE_DIR)
        


        try:
            raw_path = self.validation_config.raw_data_path
            logging.info("Inside try block")

            if not os.path.exists(raw_path):
                logging.info("Inside try block line-2")
                raise FileNotFoundError(f"Raw data not found at {raw_path}")

            logging.info("Inside try block line-3")

            df = pd.read_csv(raw_path)
            logging.info("Inside try block line-4")
            logging.info(f"Raw dataset loaded with shape: {df.shape}")

            report_lines = []

            # Basic Info
            report_lines.append(f"Dataset Shape: {df.shape}")

            # Missing Values
            null_counts = df.isnull().sum()
            report_lines.append("\nMissing Values:")
            report_lines.append(str(null_counts))

            # Duplicate Rows
            duplicates = df.duplicated().sum()
            report_lines.append(f"\nDuplicate Rows: {duplicates}")

            # Data Types
            report_lines.append("\nData Types:")
            report_lines.append(str(df.dtypes))

            # Basic Statistics
            report_lines.append("\nBasic Statistics:")
            report_lines.append(str(df.describe(include='all')))

            # Save validation report
            os.makedirs(os.path.dirname(self.validation_config.validation_report_path), exist_ok=True)

            with open(self.validation_config.validation_report_path, "w") as f:
                for line in report_lines:
                    f.write(line + "\n")

            logging.info("Structured validation report generated successfully")

            return self.validation_config.validation_report_path

        except Exception as e:
            logging.error("Error occurred during Structured Data Validation")
            raise CustomException(e, sys)