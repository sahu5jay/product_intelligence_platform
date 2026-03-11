import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.utils import save_object
from src.shared_utils.constants import (
    DEVICE,
    DEFAULT_LATENT_DIM,
    DEFAULT_FEATURE_MAPS_GEN,
    DEFAULT_FEATURE_MAPS_DISC,
    DEFAULT_LR,
    DEFAULT_BETA1,
    DEFAULT_BETA2,
    DEFAULT_EPOCHS,
    DEFAULT_CHANNELS,
    GENERATED_IMAGES_DIR,
    GENERATOR_MODEL_PATH,
    DISCRIMINATOR_MODEL_PATH,
    CHECKPOINTS_DIR
)

from src.gan_module.models.generator import Generator
from src.gan_module.models.discriminator import Discriminator


class GANTrainer:

    def __init__(self, config=None):
        try:
            logging.info("Initializing GAN Trainer")

            self.device = DEVICE
            self.latent_dim = config.get("noise_dim", DEFAULT_LATENT_DIM) if config else DEFAULT_LATENT_DIM
            self.channels = config.get("channels", DEFAULT_CHANNELS) if config else DEFAULT_CHANNELS
            self.epochs = config.get("epochs", DEFAULT_EPOCHS) if config else DEFAULT_EPOCHS
            self.lr = config.get("learning_rate", DEFAULT_LR) if config else DEFAULT_LR
            self.beta1 = config.get("beta1", DEFAULT_BETA1) if config else DEFAULT_BETA1
            self.beta2 = config.get("beta2", DEFAULT_BETA2) if config else DEFAULT_BETA2
            self.feature_maps_gen = config.get("feature_maps_gen", DEFAULT_FEATURE_MAPS_GEN) if config else DEFAULT_FEATURE_MAPS_GEN
            self.feature_maps_disc = config.get("feature_maps_disc", DEFAULT_FEATURE_MAPS_DISC) if config else DEFAULT_FEATURE_MAPS_DISC

            # -----------------------------
            # Initialize Models
            # -----------------------------
            logging.info("Loading Generator")
            self.generator = Generator(
                latent_dim=self.latent_dim,
                channels=self.channels,
                feature_maps=self.feature_maps_gen
            ).to(self.device)

            logging.info("Loading Discriminator")
            self.discriminator = Discriminator(
                channels=self.channels,
                feature_maps=self.feature_maps_disc
            ).to(self.device)

            # -----------------------------
            # Loss & Optimizers
            # -----------------------------
            self.criterion = nn.BCEWithLogitsLoss()
            self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
            self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

            # Mixed precision
            self.scaler = GradScaler()

            # Ensure directories exist
            GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            GENERATOR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            DISCRIMINATOR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

            logging.info("GAN Trainer initialized successfully")

        except Exception as e:
            logging.error("Error initializing GAN Trainer")
            raise CustomException(e, sys)

    def train(self, dataloader):
        try:
            logging.info("Starting GAN Training")
            torch.backends.cudnn.benchmark = True  

            for epoch in range(self.epochs):
                logging.info(f"Starting Epoch {epoch+1}/{self.epochs}")

                for batch_idx, (real_images,) in enumerate(dataloader):
                    real_images = real_images.to(self.device, non_blocking=True)
                    batch_size = real_images.size(0)

                    real_labels = torch.ones(batch_size, device=self.device) * 0.9
                    fake_labels = torch.zeros(batch_size, device=self.device)

                    # -------------------------
                    # Train Discriminator
                    # -------------------------
                    self.optimizer_d.zero_grad()
                    noise = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)

                    with autocast():
                        fake_images = self.generator(noise)
                        real_output = self.discriminator(real_images)
                        loss_real = self.criterion(real_output, real_labels)
                        fake_output = self.discriminator(fake_images.detach())
                        loss_fake = self.criterion(fake_output, fake_labels)
                        loss_d = loss_real + loss_fake

                    self.scaler.scale(loss_d).backward()
                    self.scaler.step(self.optimizer_d)
                    self.scaler.update()

                    # -------------------------
                    # Train Generator
                    # -------------------------
                    self.optimizer_g.zero_grad()
                    with autocast():
                        output = self.discriminator(fake_images)
                        loss_g = self.criterion(output, real_labels)

                    self.scaler.scale(loss_g).backward()
                    self.scaler.step(self.optimizer_g)
                    self.scaler.update()

                    if batch_idx % 100 == 0:
                        logging.info(
                            f"Epoch [{epoch+1}/{self.epochs}] "
                            f"Batch [{batch_idx}] "
                            f"Loss_D: {loss_d.item():.4f} "
                            f"Loss_G: {loss_g.item():.4f}"
                        )

                # -------------------------
                # Generate sample images & save checkpoints
                # -------------------------
                noise = torch.randn(16, self.latent_dim, 1, 1, device=self.device)
                with torch.no_grad():
                    fake_images = self.generator(noise)

                # Save sample images
                save_path = GENERATED_IMAGES_DIR / f"epoch_{epoch+1}.png"
                save_image(fake_images, save_path, normalize=True)

                # Save epoch checkpoints
                torch.save(self.generator.state_dict(), CHECKPOINTS_DIR / f"generator_epoch_{epoch+1}.pt")
                torch.save(self.discriminator.state_dict(), CHECKPOINTS_DIR / f"discriminator_epoch_{epoch+1}.pt")

                logging.info(f" Epoch {epoch+1}: Sample images and checkpoints saved")

            # -------------------------
            # Save Final Models
            # -------------------------
            logging.info("Saving final trained GAN models")

            save_object(
                file_path=str(GENERATOR_MODEL_PATH),
                obj=self.generator.state_dict()
            )

            save_object(
                file_path=str(DISCRIMINATOR_MODEL_PATH),
                obj=self.discriminator.state_dict()
            )

            logging.info(f"Generator model saved at {GENERATOR_MODEL_PATH}")
            logging.info(f"Discriminator model saved at {DISCRIMINATOR_MODEL_PATH}")
            logging.info(" Final models saved successfully")

        except Exception as e:
            logging.error("Error occurred during GAN training")
            raise CustomException(e, sys)