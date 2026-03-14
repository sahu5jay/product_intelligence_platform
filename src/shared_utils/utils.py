import os
import numpy as np
import pickle
import joblib
import json
from pathlib import Path
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
import sys

# -------------------------------
# Save Python object to file
# -------------------------------
def save_json(file_path: str, data: dict):
    """
    Save dictionary to JSON file.
    """

    try:
        file_path = Path(file_path)

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # If user mistakenly passes a directory, fix automatically
        if file_path.suffix == "":
            file_path = file_path / "output.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        logging.info(f"JSON saved at {file_path}")

    except Exception as e:
        logging.error(f"Error saving JSON at {file_path}")
        raise CustomException(e, sys)

# -------------------------------
# Load Python object from file
# -------------------------------
def load_object(file_path: str):
    """
    Load a Python object from a pickle file.
    """

    try:

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as file_obj:
            obj = joblib.load(file_obj)

        logging.info(f"Object loaded successfully from {file_path}")

        return obj

    except Exception as e:

        logging.error(f"Error loading object from {file_path}")

        raise CustomException(e, sys)

# -------------------------------
# Save dictionary to JSON file
# -------------------------------
def save_json(file_path, data: dict):
    """
    Save dictionary to JSON file.
    """

    try:
        file_path = Path(file_path)

        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        logging.info(f"JSON saved at {file_path}")

    except Exception as e:
        logging.error(f"Error saving JSON at {file_path}")
        raise CustomException(e, sys)

# -------------------------------
# convert csv to numpy array
# -------------------------------
def save_numpy(file_path: str, array: np.ndarray):
    """
    Save a NumPy array to a file.
    Creates parent directories if they don't exist.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, array)
        logging.info(f"NumPy array saved at {file_path}")
    except Exception as e:
        logging.error(f"Error saving NumPy array at {file_path}")
        raise CustomException(e, sys)

# -------------------------------
# Load dictionary from JSON file
# -------------------------------
def load_json(file_path: str) -> dict:
    """
    Load dictionary from JSON file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info(f"JSON loaded from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading JSON from {file_path}")
        raise CustomException(e, sys)

# -------------------------------
# Ensure directory exists
# -------------------------------
def ensure_dir(directory: str):
    """
    Create directory if it doesn't exist.
    """
    try:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Directory ready: {directory}")
    except Exception as e:
        logging.error(f"Error creating directory: {directory}")
        raise CustomException(e, sys)

# -------------------------------
# Flatten nested list
# -------------------------------
def flatten_list(nested_list):
    """
    Flatten a list of lists into a single list.
    """
    return [item for sublist in nested_list for item in sublist]

# -------------------------------
# Safe division
# -------------------------------
def safe_divide(a, b, default=0):
    """
    Divide a by b safely; returns default if division by zero occurs.
    """
    try:
        return a / b if b != 0 else default
    except Exception:
        return default

def save_object(file_path: str, obj: object):

    try:
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)

        joblib.dump(obj, file_path)

        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Error saving object at {file_path}")
        raise CustomException(e, sys)