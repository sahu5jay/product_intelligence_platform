# src/gan_module/components/image_ingestion.py

import os
import sys
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.config_loader import load_config

# ===============================
# Load config
# ===============================
BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "src" / "gan_module" / "config.yaml"
config = load_config(CONFIG_PATH)

@dataclass
class ImageIngestionConfig:
    raw_data_path: str = str(BASE_DIR / config["image_ingestion"]["raw_data_path"])
    processed_data_path: str = str(BASE_DIR / config["image_ingestion"]["processed_data_path"])

class ImageIngestion:

    def __init__(self):
        self.ingestion_config = ImageIngestionConfig()

    def initiate_image_ingestion(self):
        logging.info("Image Data Ingestion Started")

        try:
            raw_path = self.ingestion_config.raw_data_path

            if not os.path.exists(raw_path):
                raise FileNotFoundError(f"Dataset not found: {raw_path}")

            df = pd.read_csv(raw_path)
            logging.info(f"Dataset loaded with shape {df.shape}")

            # Extract pixel values (all columns except first if it is ID)
            images = df.iloc[:, 1:].values if df.shape[1] > 1 else df.values

            # Normalize to 0-1
            images = images / 255.0

            # Reshape for CNN: (N, C, H, W)
            images = images.reshape(-1, 1, 28, 28)

            os.makedirs(os.path.dirname(self.ingestion_config.processed_data_path), exist_ok=True)
            np.save(self.ingestion_config.processed_data_path, images)

            logging.info(f"Processed images saved at {self.ingestion_config.processed_data_path}")

            return self.ingestion_config.processed_data_path

        except Exception as e:
            logging.error("Error in Image Data Ingestion")
            raise CustomException(e, sys)