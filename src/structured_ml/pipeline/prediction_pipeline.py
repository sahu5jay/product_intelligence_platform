# src/structured_ml/pipeline/prediction_pipeline.py

import sys
import pandas as pd
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.utils import load_object
from src.shared_utils.constants import STRUCTURED_MODEL_PATH, PREPROCESSOR_PATH


class PredictPipeline:
    """
    Pipeline to handle loading of pre-trained model and preprocessor
    and generating predictions for new data.
    """

    def __init__(self, model_path: str = STRUCTURED_MODEL_PATH, preprocessor_path: str = PREPROCESSOR_PATH):
        try:
            logging.info("Loading trained model and preprocessor")
            self.model = load_object(model_path)
            self.preprocessor = load_object(preprocessor_path)
            logging.info("Model and preprocessor loaded successfully")
        except Exception as e:
            logging.error("Error loading model or preprocessor")
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        """
        Accepts a DataFrame of features, applies preprocessing, and returns predictions
        """
        try:
            logging.info("Transforming features using preprocessor")
            data_scaled = self.preprocessor.transform(features)

            logging.info("Generating predictions")
            predictions = self.model.predict(data_scaled)

            return predictions

        except Exception as e:
            logging.error("Error during prediction")
            raise CustomException(e, sys)


class CustomData:
    """
    Utility class to convert user input into a DataFrame compatible with the pipeline
    """

    def __init__(self,
                 GrLivArea: float,
                 OverallQual: float,
                 YearBuilt: float,
                 TotalBsmtSF: float,
                 GarageCars: float,
                 Neighborhood: str,
                 ExterQual: str,
                 KitchenQual: str):

        self.GrLivArea = GrLivArea
        self.OverallQual = OverallQual
        self.YearBuilt = YearBuilt
        self.TotalBsmtSF = TotalBsmtSF
        self.GarageCars = GarageCars
        self.Neighborhood = Neighborhood
        self.ExterQual = ExterQual
        self.KitchenQual = KitchenQual

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Convert the input features into a Pandas DataFrame
        """
        try:
            logging.info("Creating DataFrame from user input")

            data_dict = {
                "GrLivArea": [self.GrLivArea],
                "OverallQual": [self.OverallQual],
                "YearBuilt": [self.YearBuilt],
                "TotalBsmtSF": [self.TotalBsmtSF],
                "GarageCars": [self.GarageCars],
                "Neighborhood": [self.Neighborhood],
                "ExterQual": [self.ExterQual],
                "KitchenQual": [self.KitchenQual]
            }

            df = pd.DataFrame(data_dict)
            logging.info("DataFrame created successfully")

            return df

        except Exception as e:
            logging.error("Exception occurred while creating DataFrame from user input")
            raise CustomException(e, sys)