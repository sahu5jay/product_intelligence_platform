import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


BASE_DIR = Path(__file__).resolve().parents[3]


@dataclass
class ImageIngestionConfig:

    raw_data_path: str = str(BASE_DIR / "notebook" / "data" / "images" / "fassion_image.csv")

    processed_data_path: str = str(BASE_DIR / "artifacts" / "gan" / "processed_images.npy")


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

            # Extract pixel values
            images = df.iloc[:, 1:].values

            # Normalize
            images = images / 255.0

            # Reshape to CNN format
            images = images.reshape(-1, 1, 28, 28)

            os.makedirs(
                os.path.dirname(self.ingestion_config.processed_data_path),
                exist_ok=True
            )

            np.save(self.ingestion_config.processed_data_path, images)

            logging.info(f"Processed images saved at {self.ingestion_config.processed_data_path}")

            return self.ingestion_config.processed_data_path

        except Exception as e:

            logging.error("Error in Image Data Ingestion")

            raise CustomException(e, sys)