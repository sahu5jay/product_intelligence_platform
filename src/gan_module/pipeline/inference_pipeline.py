import sys
import os
import torch
import uuid
from torchvision.utils import save_image

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.constants import (
    GENERATOR_MODEL_PATH,
    GENERATED_IMAGES_DIR,
    DEFAULT_LATENT_DIM,
    DEVICE
)

from src.gan_module.models.generator import Generator


class GANInferencePipeline:
    """
    Pipeline to generate images using trained GAN model
    """

    def __init__(self):

        try:
            logging.info("Initializing GAN Inference Pipeline")

            self.device = DEVICE
            self.latent_dim = DEFAULT_LATENT_DIM

            # Load Generator Model
            self.generator = Generator().to(self.device)

            if not os.path.exists(GENERATOR_MODEL_PATH):
                raise Exception("Generator model not found")

            self.generator.load_state_dict(
                torch.load(GENERATOR_MODEL_PATH, map_location=self.device)
            )

            self.generator.eval()

            logging.info("Generator model loaded successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def generate_images(self, label: str, num_images: int):

        """
        Generate images based on selected label
        """

        try:

            logging.info(f"Generating {num_images} images for label {label}")

            os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

            image_paths = []

            for i in range(num_images):

                noise = torch.randn(1, self.latent_dim, 1, 1).to(self.device)

                with torch.no_grad():
                    fake_image = self.generator(noise)

                file_name = f"{label}_{uuid.uuid4().hex}.png"

                save_path = os.path.join(GENERATED_IMAGES_DIR, file_name)

                save_image(fake_image, save_path, normalize=True)

                image_paths.append(f"/static/generated/{file_name}")

            logging.info("Images generated successfully")

            return image_paths

        except Exception as e:
            logging.error("Error in image generation")
            raise CustomException(e, sys)