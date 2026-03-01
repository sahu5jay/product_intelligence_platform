# src/structured_ml/components/data_transformation.py

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.shared_utils.exception import CustomException
from src.shared_utils.logger import logging
from src.shared_utils.save_object import save_object


# =====================================
# Config
# =====================================

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "structured", "preprocessor.pkl"
    )


# =====================================
# Data Transformation Class
# =====================================

class DataTransformation:

    def __init__(self):
        self.config = DataTransformationConfig()

    # ------------------------------------------------
    # Automatically detect column types
    # ------------------------------------------------
    def detect_column_types(self, df):

        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        logging.info(f"Detected Numerical Columns: {numerical_cols}")
        logging.info(f"Detected Categorical Columns: {categorical_cols}")

        return numerical_cols, categorical_cols


    # ------------------------------------------------
    # Create Preprocessor
    # ------------------------------------------------
    def get_preprocessor(self, numerical_cols, categorical_cols):

        try:
            logging.info("Creating preprocessing pipelines")

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ordinalencoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1
                        )
                    ),
                    ("scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    # ------------------------------------------------
    # Initiate Transformation
    # ------------------------------------------------
    def initiate_data_transformation(
        self,
        train_path: str,
        test_path: str,
        target_column: str = "SalePrice"
    ):

        try:
            logging.info("Reading train and test data")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Train shape: {train_df.shape}")
            logging.info(f"Test shape: {test_df.shape}")

            # -----------------------------
            # Split features & target FIRST
            # -----------------------------
            if target_column not in train_df.columns:
                raise Exception(f"{target_column} not found in train dataset")

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            # Test may or may not contain target
            if target_column in test_df.columns:
                X_test = test_df.drop(columns=[target_column])
                y_test = test_df[target_column]
            else:
                X_test = test_df
                y_test = None

            # ---------------------------------
            # Detect column types on FEATURES
            # ---------------------------------
            numerical_cols, categorical_cols = self.detect_column_types(X_train)

            # ---------------------------------
            # Create preprocessor
            # ---------------------------------
            preprocessor = self.get_preprocessor(
                numerical_cols, categorical_cols
            )

            # ---------------------------------
            # Apply transformation
            # ---------------------------------
            logging.info("Applying preprocessing")

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            logging.info("Transformation completed successfully")

            # ---------------------------------
            # Combine features with target
            # ---------------------------------
            train_arr = np.c_[X_train_arr, np.array(y_train)]

            if y_test is not None:
                test_arr = np.c_[X_test_arr, np.array(y_test)]
            else:
                test_arr = X_test_arr

            # ---------------------------------
            # Save preprocessor
            # ---------------------------------
            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path), exist_ok=True)

            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("Preprocessor saved successfully")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            logging.error("Error occurred in data transformation")
            raise CustomException(e, sys)