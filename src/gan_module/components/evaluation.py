# src/gan_module/components/evaluation.py

import os
import sys
from pathlib import Path
import torch
import numpy as np
from torchvision.utils import save_image

from dataclasses import dataclass
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.config_loader import load_config

from src.gan_module.components.generator import Generator

# ===============================
# Load Config
# ===============================
BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "src" / "gan_module" / "config.yaml"
config = load_config(CONFIG_PATH)

# ===============================
# Evaluation Config Dataclass
# ===============================
@dataclass
class GANEvaluationConfig:
    latent_dim: int = config["gan_model"]["noise_dim"]
    generator_path: str = str(BASE_DIR / config["model_paths"]["generator_path"])
    output_dir: str = str(BASE_DIR / config["generated_images"]["output_dir"])
    num_samples: int = 64  # Number of images to generate
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================
# GAN Evaluation Class
# ===============================
class GANEvaluator:

    def __init__(self):
        try:
            self.config = GANEvaluationConfig()
            os.makedirs(self.config.output_dir, exist_ok=True)

            # Load Generator
            self.generator = Generator(latent_dim=self.config.latent_dim)
            self.generator.load_state_dict(torch.load(self.config.generator_path, map_location=self.config.device))
            self.generator.to(self.config.device)
            self.generator.eval()

            logging.info("Generator loaded for evaluation")

        except Exception as e:
            logging.error("Error initializing GANEvaluator")
            raise CustomException(e, sys)

    def generate_images(self, num_samples=None, save=True):
        try:
            num_samples = num_samples or self.config.num_samples

            logging.info(f"Generating {num_samples} images using the GAN")

            noise = torch.randn(num_samples, self.config.latent_dim, device=self.config.device)
            with torch.no_grad():
                fake_images = self.generator(noise)

            # Scale images to [0,1] for saving
            fake_images = (fake_images + 1) / 2.0

            if save:
                for idx, img in enumerate(fake_images):
                    save_path = Path(self.config.output_dir) / f"generated_{idx}.png"
                    save_image(img, save_path)

                logging.info(f"{num_samples} generated images saved at {self.config.output_dir}")

            return fake_images

        except Exception as e:
            logging.error("Error during image generation")
            raise CustomException(e, sys)