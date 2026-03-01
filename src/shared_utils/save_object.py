# src/structured_ml/utils/save_object.py

import os
import sys
import pickle
from src.shared_utils.exception import CustomException
from src.shared_utils.logger import logging


def save_object(file_path: str, obj: object):
    """
    Save a Python object to a file using pickle.

    Args:
        file_path (str): Path where the object will be saved
        obj (object): Python object to save
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Error saving object at {file_path}")
        raise CustomException(e, sys)