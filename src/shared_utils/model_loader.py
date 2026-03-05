import sys
import joblib

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


def load_object(file_path):
    try:
        logging.info(f"Loading object from {file_path}")

        obj = joblib.load(file_path)

        logging.info("Object loaded successfully")

        return obj

    except Exception as e:
        logging.info("Exception occurred in load_object function")
        raise CustomException(e, sys)