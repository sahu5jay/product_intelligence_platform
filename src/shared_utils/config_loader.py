import sys
import yaml
from pathlib import Path

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


def load_config(config_path):
    """
    Load YAML configuration file.

    Args:
        config_path (str | Path): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """

    try:
        # Convert to Path object
        config_path = Path(config_path).resolve()

        logging.info(f"Loading config file from: {config_path}")

        # Check file existence
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load YAML
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        # Validate config
        if config is None:
            raise ValueError("Config file is empty or invalid YAML")

        logging.info("Config file loaded successfully")

        return config

    except Exception as e:
        logging.error("Error while loading config file")
        raise CustomException(e, sys)