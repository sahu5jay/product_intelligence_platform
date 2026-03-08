import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.utils import save_object
from src.shared_utils.config_loader import load_config
from src.shared_utils.constants import PREPROCESSOR_PATH

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
        self.preprocessor_path = PREPROCESSOR_PATH
        self.numeric_columns = NUMERIC_COLUMNS
        self.categorical_columns = CATEGORICAL_COLUMNS
        self.target_column = TARGET_COLUMN

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

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Read train/test CSVs, apply preprocessing, and return transformed arrays
        concatenated with the target variable.
        """
        try:
            logging.info(f"Reading training data from {train_path}")
            train_df = pd.read_csv(train_path)
            logging.info(f"Reading testing data from {test_path}")
            test_df = pd.read_csv(test_path)

            # Check all required columns exist
            for col in self.numeric_columns + self.categorical_columns:
                if col not in train_df.columns:
                    raise ValueError(f"Column '{col}' missing in training data")
                if col not in test_df.columns:
                    raise ValueError(f"Column '{col}' missing in testing data")
            if self.target_column not in train_df.columns or self.target_column not in test_df.columns:
                raise ValueError(f"Target column '{self.target_column}' missing in train or test data")

            logging.info("Building preprocessor")
            preprocessor = self.build_preprocessor()

            # Split features and target
            X_train = train_df.drop(columns=[self.target_column])
            y_train = train_df[self.target_column]
            X_test = test_df.drop(columns=[self.target_column])
            y_test = test_df[self.target_column]

            # Fit-transform train, transform test
            logging.info("Fitting preprocessor on training data")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Concatenate features + target for train/test
            train_array = np.hstack([X_train_transformed, y_train.values.reshape(-1, 1)])
            test_array = np.hstack([X_test_transformed, y_test.values.reshape(-1, 1)])

            # Save preprocessor
            save_object(self.preprocessor_path, preprocessor)
            logging.info(f"Preprocessor saved at {self.preprocessor_path}")

            return train_array, test_array, self.preprocessor_path

        except Exception as e:
            logging.error("Error during data transformation")
            raise CustomException(e, sys)