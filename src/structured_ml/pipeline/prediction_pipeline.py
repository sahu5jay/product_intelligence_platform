import sys
import os
import pandas as pd

from src.shared_utils.exception import CustomException
from src.shared_utils.logger import logging
from src.shared_utils.model_loader import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:
            model_path = os.path.join("artifacts", "structured", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "structured", "preprocessor.pkl")

            logging.info("Loading model and preprocessor")

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)

            predictions = model.predict(data_scaled)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
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

    def get_data_as_dataframe(self):
        try:
            logging.info("Creating dataframe from user input")

            custom_data_input_dict = {
                "GrLivArea": [self.GrLivArea],
                "OverallQual": [self.OverallQual],
                "YearBuilt": [self.YearBuilt],
                "TotalBsmtSF": [self.TotalBsmtSF],
                "GarageCars": [self.GarageCars],
                "Neighborhood": [self.Neighborhood],
                "ExterQual": [self.ExterQual],
                "KitchenQual": [self.KitchenQual]
            }

            df = pd.DataFrame(custom_data_input_dict)

            logging.info("Dataframe created successfully")

            return df

        except Exception as e:
            logging.info("Exception occurred in CustomData class")
            raise CustomException(e, sys)