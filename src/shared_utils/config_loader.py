import sys
import yaml
from pathlib import Path

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


def load_config(config_path: Path):

    try:
        logging.info(f"Loading config file from: {config_path}")

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        logging.info("Config file loaded successfully")

        return config

    except Exception as e:
        logging.error("Error while loading config file")
        raise CustomException(e, sys)