import os
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from PIL import Image

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


# ==========================================
# Configuration
# ==========================================

@dataclass
class ImageTransformationConfig:

    transformed_dir: str = str(
        Path("artifacts") / "gan" / "transformed_images"
    )

    image_size: tuple = (128, 128)


# ==========================================
# Image Transformation Component
# ==========================================

class ImageTransformation:

    def __init__(self):
        self.config = ImageTransformationConfig()

    def initiate_image_transformation(self, input_path: str):

        logging.info("Image Transformation Started")

        try:

            input_path = Path(input_path)

            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")

            # load numpy array
            images = np.load(input_path)

            output_dir = Path(self.config.transformed_dir)
            os.makedirs(output_dir, exist_ok=True)

            transformed_count = 0

            for idx, img in enumerate(images):

                img = img.squeeze()

                pil_image = Image.fromarray((img * 255).astype(np.uint8))

                pil_image = pil_image.resize(self.config.image_size)

                save_path = output_dir / f"image_{idx}.png"

                pil_image.save(save_path)

                transformed_count += 1

            logging.info(f"{transformed_count} images transformed successfully")

            return str(output_dir)

        except Exception as e:

            logging.error("Error occurred during Image Transformation")

            raise CustomException(e, sys)