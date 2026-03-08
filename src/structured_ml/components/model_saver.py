# src/structured_ml/components/model_saver.py
import sys
from pathlib import Path
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.utils import save_object
from src.shared_utils.config_loader import load_config

# Load config
CONFIG_PATH = Path(__file__).resolve().parents[3] / "src/structured_ml/config.yaml"
config = load_config(CONFIG_PATH)
MODEL_SAVE_PATH = Path(config["model_saver"]["save_path"])


class ModelSaver:
    """
    Saves a trained model to the path specified in config.yaml
    """
    def __init__(self, save_path: Path = MODEL_SAVE_PATH):
        self.save_path = save_path

    def save_model(self, model):
        """
        Save the trained model object to disk
        """
        try:
            logging.info(f"Saving model to {self.save_path}")
            save_object(self.save_path, model)
            logging.info(f"Model successfully saved at {self.save_path}")
            return str(self.save_path)

        except Exception as e:
            logging.error("Error saving model")
            raise CustomException(e, sys)