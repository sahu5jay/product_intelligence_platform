import sys
import torch
from torchvision.utils import save_image

from src.gan_module.models.generator import Generator
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.utils import load_object
from src.shared_utils.constants import (
DEVICE,
DEFAULT_LATENT_DIM,
DEFAULT_CHANNELS,
DEFAULT_FEATURE_MAPS_GEN,
MAX_GENERATED_IMAGES,
GENERATOR_MODEL_PATH,
GENERATED_IMAGES_DIR,
GAN_SAMPLE_GRID
)

class GANEvaluation:
    """
    Evaluate trained GAN Generator and generate sample images
    """

    def __init__(self):

        try:

            self.device = DEVICE
            self.latent_dim = DEFAULT_LATENT_DIM

            logging.info("Initializing GAN Evaluation")

            # Initialize generator architecture
            self.generator = Generator(
                latent_dim=DEFAULT_LATENT_DIM,
                channels=DEFAULT_CHANNELS,
                feature_maps=DEFAULT_FEATURE_MAPS_GEN
            ).to(self.device)

            # Check if model exists
            if not GENERATOR_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Generator model not found at {GENERATOR_MODEL_PATH}"
                )

            # Load state_dict correctly
            logging.info(f"Loading generator model from {GENERATOR_MODEL_PATH}")

            state_dict = load_object(GENERATOR_MODEL_PATH)

            self.generator.load_state_dict(state_dict)

            self.generator.to(self.device)
            self.generator.eval()

            logging.info("Generator model loaded successfully")


        except Exception as e:
            raise CustomException(e, sys)


    def generate_images(self):

        """
        Generate images using trained Generator
        """

        try:

            logging.info("Generating images using trained GAN")

            noise = torch.randn(
                MAX_GENERATED_IMAGES,
                self.latent_dim,
                1,
                1,
                device=self.device
            )

            with torch.no_grad():
                fake_images = self.generator(noise)

            # Save each generated image
            for i, img in enumerate(fake_images):

                save_path = GENERATED_IMAGES_DIR / f"generated_{i}.png"

                save_image(
                    img,
                    save_path,
                    normalize=True
                )

            logging.info(
                f"{MAX_GENERATED_IMAGES} images saved to {GENERATED_IMAGES_DIR}"
            )

            # Save grid image for evaluation
            save_image(
                fake_images,
                GAN_SAMPLE_GRID,
                normalize=True,
                nrow=5
            )

            logging.info(f"Evaluation grid saved at {GAN_SAMPLE_GRID}")

        except Exception as e:

            logging.error("Error during GAN evaluation")

            raise CustomException(e, sys)

