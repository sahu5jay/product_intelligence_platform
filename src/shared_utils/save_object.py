# src/structured_ml/utils/save_object.py

import os
import sys
import joblib

from src.shared_utils.exception import CustomException
from src.shared_utils.logger import logging


def save_object(file_path: str, obj: object):

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        joblib.dump(obj, file_path)

        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Error saving object at {file_path}")
        raise CustomException(e, sys)