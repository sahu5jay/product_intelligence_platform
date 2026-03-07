import os
import sys
import torch
import torchvision.utils as vutils

from src.gan_module.components.generator import Generator
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


class GANEvaluator:

    def __init__(self, model_path,
                 output_dir="frontend/static/generated_images",
                 noise_dim=100):

        self.model_path = model_path
        self.output_dir = output_dir
        self.noise_dim = noise_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.output_dir, exist_ok=True)

        # Load generator once
        self.generator = self.load_generator()


    def load_generator(self):

        try:

            logging.info("Loading trained Generator model")

            generator = Generator().to(self.device)

            generator.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )

            generator.eval()

            logging.info("Generator model loaded successfully")

            return generator

        except Exception as e:
            raise CustomException(e, sys)


    def generate_images(self, num_images):

        try:

            logging.info(f"Generating {num_images} images")

            noise = torch.randn(num_images, self.noise_dim).to(self.device)

            with torch.no_grad():
                fake_images = self.generator(noise)

            image_paths = []

            for i in range(num_images):

                file_name = f"generated_{i}.png"

                full_path = os.path.join(self.output_dir, file_name)

                vutils.save_image(fake_images[i], full_path, normalize=True)

                image_paths.append(f"/static/generated_images/{file_name}")

            logging.info("Images generated successfully")

            return image_paths

        except Exception as e:
            raise CustomException(e, sys)