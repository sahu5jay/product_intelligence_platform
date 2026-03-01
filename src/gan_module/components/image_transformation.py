# src/gan_module/components/image_transformation.py

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
import numpy as np

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException

# Base directory of your project
BASE_DIR = Path(__file__).resolve().parents[3]


@dataclass
class ImageTransformationConfig:
    # Folder containing raw images
    input_image_dir: str = str(BASE_DIR / "notebook" / "data" / "images" / "fashion_image.csv")
    # Folder where transformed images will be saved
    transformed_dir: str = str(BASE_DIR / "artifacts" / "transformed_images")
    # Target image size for GAN
    image_size: tuple = (128, 128)


class ImageTransformation:
    def __init__(self):
        self.config = ImageTransformationConfig()

    def initiate_image_transformation(self):
        logging.info("Image Transformation Started")

        try:
            input_dir = Path(self.config.input_image_dir)
            output_dir = Path(self.config.transformed_dir)

            # Check if input directory exists
            if not input_dir.exists() or not input_dir.is_dir():
                raise FileNotFoundError(f"Input image directory not found at {input_dir}")

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Output directory created at {output_dir}")

            # Get list of image files
            image_files = list(input_dir.glob("*.*"))
            transformed_count = 0

            for img_path in image_files:
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    # Open and convert to RGB
                    image = Image.open(img_path).convert("RGB")
                    # Resize image
                    image = image.resize(self.config.image_size)
                    # Normalize pixel values (0-1)
                    image_array = np.array(image) / 255.0
                    # Save transformed image
                    save_path = output_dir / img_path.name
                    Image.fromarray((image_array * 255).astype(np.uint8)).save(save_path)
                    transformed_count += 1

            logging.info(f"Transformed {transformed_count} images successfully")
            return str(output_dir)

        except Exception as e:
            logging.error("Error occurred during Image Transformation")
            raise CustomException(e, sys)