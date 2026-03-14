import sys
import pandas as pd
import numpy as np
from pathlib import Path

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
# from src.shared_utils.constants import PROCESSED_DATA_PATH, GAN_RAW_DATA_PATH


class ImageIngestion:

    def __init__(self, gan_data_path: Path, raw_data_path: Path, processed_data_path: Path):
        self.gan_data_path = gan_data_path
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        # self.data_path = DATA_PATH

    def initiate_image_ingestion(self):

        logging.info("Starting Image Ingestion")

        try:

            if not self.raw_data_path.exists():
                raise FileNotFoundError(f"Dataset not found: {self.raw_data_path}")

            logging.info(f"Reading dataset from: {self.raw_data_path}")

            df = pd.read_csv(self.gan_data_path)

            logging.info(f"Dataset shape: {df.shape}")

            # Create directories BEFORE saving files
            logging.info("Saving Raw Data")
            self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Saving Raw Data {self.raw_data_path}")
            # self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)

            # Save raw dataset
            df.to_csv(self.raw_data_path, index=False)
            logging.info(f"Raw dataset saved at: {self.raw_data_path}")

            # Remove label column
            if "label" in df.columns:
                logging.info("Dropping label column")
                df = df.drop(columns=["label"])

            # Convert dataframe → numpy
            images = df.values.astype("float32")

            logging.info(f"Converted dataset to numpy array: {images.shape}")

            # Save processed dataset
            np.save(self.processed_data_path, images)

            logging.info(f"Processed dataset saved at: {self.processed_data_path}")

            return self.processed_data_path

        except Exception as e:
            logging.error("Error occurred during image ingestion")
            raise CustomException(e, sys)