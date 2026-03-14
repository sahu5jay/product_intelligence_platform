import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.structured_ml.components.data_validation import DataValidation
from sklearn.model_selection import train_test_split

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.utils import save_object
from src.shared_utils.config_loader import load_config
from src.shared_utils.constants import PROCESSED_STRUCTURED_DATA, TRIMED_DATA_PATH,TRIMED_VALIDATION_REPORT_PATH, PREPROCESSOR_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH
from src.shared_utils.constants import ARRAY_DIR, TRAIN_ARRAY_PATH, TEST_ARRAY_PATH

# ------------------------------
# Load config
# ------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "src/structured_ml/config.yaml"
config = load_config(CONFIG_PATH)

NUMERIC_COLUMNS = config["data_transformation"]["numerical_columns"]
CATEGORICAL_COLUMNS = config["data_transformation"]["categorical_columns"]
TARGET_COLUMN = "SalePrice"


class DataTransformation:
    """
    Handles preprocessing for structured ML data:
    - Imputes missing values
    - Scales numerical features
    - One-hot encodes categorical features
    - Returns transformed train and test arrays ready for modeling
    """

    def __init__(self):
        self.raw_data = PROCESSED_STRUCTURED_DATA
        self.preprocessor_path = PREPROCESSOR_PATH
        self.trimed_data_path = TRIMED_DATA_PATH
        self.trimed_validation_report = TRIMED_VALIDATION_REPORT_PATH
        self.numeric_columns = NUMERIC_COLUMNS
        self.categorical_columns = CATEGORICAL_COLUMNS
        self.target_column = TARGET_COLUMN
        self.train_data_path = TRAIN_DATA_PATH 
        self.test_data_path = TEST_DATA_PATH

    def build_preprocessor(self) -> ColumnTransformer:
        """
        Create a ColumnTransformer to handle preprocessing pipelines
        for numerical and categorical features.
        """
        try:
            # Numerical features: median imputation + standard scaling
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Categorical features: most frequent imputation + one-hot encoding
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer([
                ('num', num_pipeline, self.numeric_columns),
                ('cat', cat_pipeline, self.categorical_columns)
            ])

            logging.info("Preprocessor pipeline created successfully")
            return preprocessor

        except Exception as e:
            logging.error("Error creating preprocessor pipeline")
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        """
        Read train/test CSVs, apply preprocessing, and return transformed arrays
        concatenated with the target variable.
        """
        try:
            logging.info(f"Reading training data from {self.raw_data}")
            self.raw_df = pd.read_csv(self.raw_data)

            required_columns = self.numeric_columns + self.categorical_columns + [self.target_column]

            for col in required_columns:
                if col not in self.raw_df.columns:
                    raise ValueError(f"Column '{col}' missing in training data")

            # Keep only required columns
            raw_df = self.raw_df[required_columns]

            logging.info(f"--=-=-=-=-=-=>><><><{raw_df.columns}")

            os.makedirs(os.path.dirname(self.trimed_data_path), exist_ok=True)
            raw_df.to_csv(self.trimed_data_path, index=False)
            logging.info(f"train dataset saved at {self.trimed_data_path}")

            # logging.info(f"Reading testing data from {test_path}")
            # test_df = pd.read_csv(test_path)

            # Check all required columns exist
            for col in self.numeric_columns + self.categorical_columns:
                if col not in self.raw_df.columns:
                    raise ValueError(f"Column '{col}' missing in training data")
                # if col not in test_df.columns:
                #     raise ValueError(f"Column '{col}' missing in testing data")
            if self.target_column not in self.raw_df.columns: #or self.target_column not in test_df.columns:
                raise ValueError(f"Target column '{self.target_column}' missing in train or test data")


            train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)

            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)
            train_df.to_csv(self.train_data_path, index=False)
            logging.info(f"train dataset saved at {self.train_data_path}")

            os.makedirs(os.path.dirname(self.test_data_path), exist_ok=True)
            test_df.to_csv(self.test_data_path, index=False)
            logging.info(f"Raw dataset saved at {self.test_data_path}")

            # Split features and target
            X_train = train_df.drop(columns=[self.target_column])
            y_train = train_df[self.target_column]
            X_test = test_df.drop(columns=[self.target_column])
            y_test = test_df[self.target_column]

            logging.info("Building preprocessor")
            preprocessor = self.build_preprocessor()

            # Fit-transform train, transform test
            logging.info("Fitting preprocessor on training data")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Concatenate features + target for train/test
            train_array = np.hstack([X_train_transformed, y_train.values.reshape(-1, 1)])
            test_array = np.hstack([X_test_transformed, y_test.values.reshape(-1, 1)])

            os.makedirs(ARRAY_DIR, exist_ok=True)

            np.savetxt(TRAIN_ARRAY_PATH, train_array, delimiter=",")
            np.savetxt(TEST_ARRAY_PATH, test_array, delimiter=",")

            # Save preprocessor
            save_object(self.preprocessor_path, preprocessor)
            logging.info(f"Preprocessor saved at {self.preprocessor_path}")

            return train_array, test_array, self.preprocessor_path

        except Exception as e:
            logging.error("Error during data transformation")
            raise CustomException(e, sys)