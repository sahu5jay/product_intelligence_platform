import os
import pickle
import json
from pathlib import Path
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
import sys

# -------------------------------
# Save Python object to file
# -------------------------------
def save_object(file_path: str, obj):
    """
    Save a Python object to a file using pickle.
    Creates parent directories if not exist.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        logging.error(f"Error saving object at {file_path}")
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
            obj = pickle.load(file_obj)

        logging.info(f"Object loaded successfully from {file_path}")

        return obj

    except Exception as e:

        logging.error(f"Error loading object from {file_path}")

        raise CustomException(e, sys)

# -------------------------------
# Save dictionary to JSON file
# -------------------------------
def save_json(file_path: str, data: dict):
    """
    Save dictionary to JSON file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logging.info(f"JSON saved at {file_path}")
    except Exception as e:
        logging.error(f"Error saving JSON at {file_path}")
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