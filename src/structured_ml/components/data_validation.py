import os
import sys
import pandas as pd
from pathlib import Path
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.utils import save_json
from src.shared_utils.config_loader import load_config

# ------------------------------
# Load config and constants
# ------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "src/structured_ml/config.yaml"
config = load_config(CONFIG_PATH)

ARTIFACTS_ROOT = BASE_DIR / "artifacts"
VALIDATION_REPORT_PATH = ARTIFACTS_ROOT / "structured_ml" / "report" / "data_validation_report.json"

NUMERIC_COLUMNS = config["data_transformation"]["numerical_columns"]
CATEGORICAL_COLUMNS = config["data_transformation"]["categorical_columns"]

class DataValidation:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = Path(raw_data_path)
        self.validation_report_path = VALIDATION_REPORT_PATH

    def validate_data(self):
        """
        Validate the raw dataset:
        1. Check file exists
        2. Check for missing values
        3. Check required numerical & categorical columns
        4. Save validation report as JSON
        """
        try:
            logging.info(f"Starting data validation for {self.raw_data_path}")

            if not self.raw_data_path.exists():
                raise FileNotFoundError(f"Raw data file not found at {self.raw_data_path}")

            df = pd.read_csv(self.raw_data_path)

            # 1️⃣ Missing Values Check
            missing_values = df.isnull().sum().to_dict()

            # 2️⃣ Columns Check
            missing_numeric_cols = [col for col in NUMERIC_COLUMNS if col not in df.columns]
            missing_categorical_cols = [col for col in CATEGORICAL_COLUMNS if col not in df.columns]

            validation_report = {
                "file_checked": str(self.raw_data_path),
                "total_rows": df.shape[0],
                "total_columns": df.shape[1],
                "missing_values": missing_values,
                "missing_numeric_columns": missing_numeric_cols,
                "missing_categorical_columns": missing_categorical_cols,
                "status": "Valid" if not missing_numeric_cols and not missing_categorical_cols else "Invalid"
            }

            # Ensure report directory exists
            os.makedirs(os.path.dirname(self.validation_report_path), exist_ok=True)
            save_json(str(self.validation_report_path), validation_report)

            logging.info(f"Data validation report saved at {self.validation_report_path}")

            return validation_report

        except Exception as e:
            logging.error("Error during data validation")
            raise CustomException(e, sys)