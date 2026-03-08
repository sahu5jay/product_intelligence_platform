# src/gan_module/components/image_transformation.py

import os
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from PIL import Image

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.config_loader import load_config

# ===============================
# Load GAN config
# ===============================
BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "src" / "gan_module" / "config.yaml"
config = load_config(CONFIG_PATH)

# ===============================
# Configuration Dataclass
# ===============================
@dataclass
class ImageTransformationConfig:
    transformed_dir: str = str(BASE_DIR / config["artifacts_root"] / "transformed_images")
    image_size: tuple = (config["image_transformation"]["image_size"], config["image_transformation"]["image_size"])
    channels: int = config["image_transformation"].get("channels", 1)
    batch_size: int = config["image_transformation"].get("batch_size", 64)


# ===============================
# Image Transformation Class
# ===============================
class ImageTransformation:

    def __init__(self):
        self.config = ImageTransformationConfig()

    def initiate_image_transformation(self, input_path: str):

        logging.info("Image Transformation Started")

        try:
            input_path = Path(input_path)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")

            # Load numpy array
            images = np.load(input_path)

            output_dir = Path(self.config.transformed_dir)
            os.makedirs(output_dir, exist_ok=True)

            transformed_count = 0

            for idx, img in enumerate(images):
                img = img.squeeze()  # Remove single-dimensional entries

                # If single-channel grayscale, convert accordingly
                if self.config.channels == 1:
                    pil_image = Image.fromarray((img * 255).astype(np.uint8)).convert("L")
                else:
                    pil_image = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")

                # Resize according to config
                pil_image = pil_image.resize(self.config.image_size)

                save_path = output_dir / f"image_{idx}.png"
                pil_image.save(save_path)
                transformed_count += 1

            logging.info(f"{transformed_count} images transformed successfully")
            return str(output_dir)

        except Exception as e:
            logging.error("Error occurred during Image Transformation")
            raise CustomException(e, sys)