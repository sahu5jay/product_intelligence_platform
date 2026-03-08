# src/gan_module/components/gan_trainer.py

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F

from src.gan_module.components.generator import Generator
from src.gan_module.components.discriminator import Discriminator
from src.gan_module.components.checkpoint_manager import CheckpointManager
from src.shared_utils.save_object import save_object
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.config_loader import load_config

# ===============================
# Load Config
# ===============================
BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "src" / "gan_module" / "config.yaml"
config = load_config(CONFIG_PATH)


# ===============================
# Configuration Dataclass
# ===============================
@dataclass
class GANTrainerConfig:
    processed_data_path: str = str(BASE_DIR / config["image_ingestion"]["processed_data_path"])
    model_dir: str = str(BASE_DIR / config["model_paths"]["generator_path"]).replace("generator.pth", "")
    checkpoint_dir: str = str(BASE_DIR / "artifacts" / "gan" / "checkpoints")

    epochs: int = config["gan_training"]["epochs"]
    batch_size: int = config["image_transformation"]["batch_size"]
    lr: float = config["gan_training"]["learning_rate"]
    beta1: float = config["gan_training"]["beta1"]
    latent_dim: int = config["gan_model"]["noise_dim"]

    feature_maps_gen: int = config["gan_model"]["feature_maps_gen"]
    feature_maps_disc: int = config["gan_model"]["feature_maps_disc"]
    channels: int = config["image_transformation"]["channels"]
    image_size: int = config["image_transformation"]["image_size"]


# ===============================
# GAN Trainer Class
# ===============================
class GANTrainer:

    def __init__(self):
        try:
            self.config = GANTrainerConfig()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize Generator and Discriminator
            self.generator = Generator(
                latent_dim=self.config.latent_dim,
                channels=self.config.channels,
                feature_maps=self.config.feature_maps_gen
            ).to(self.device)
            logging.info("Generator model initialized")

            self.discriminator = Discriminator(
                channels=self.config.channels,
                feature_maps=self.config.feature_maps_disc
            ).to(self.device)
            logging.info("Discriminator model initialized")
            # Loss function
            self.loss_fn = nn.BCELoss()

            # Optimizers
            self.optimizer_G = optim.Adam(
                self.generator.parameters(),
                lr=self.config.lr,
                betas=(self.config.beta1, 0.999)
            )
            self.optimizer_D = optim.Adam(
                self.discriminator.parameters(),
                lr=self.config.lr,
                betas=(self.config.beta1, 0.999)
            )

            # Checkpoint Manager
            self.checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)

        except Exception as e:
            logging.error("Error initializing GANTrainer")
            raise CustomException(e, sys)

    # ----------------------
    # Load Image Data
    # ----------------------
    def load_data(self):
        try:
            logging.info("Loading processed image data")
            images = np.load(self.config.processed_data_path, mmap_mode="r")
            images = torch.tensor(images, dtype=torch.float32)

            # Ensure shape [N, C, H, W]
            if images.ndim == 5:  # [N, 1, 1, H, W]
                images = images.view(images.size(0), images.size(1), images.size(3), images.size(4))
            elif images.ndim == 3:  # [N, H, W]
                images = images.unsqueeze(1)

            # Resize to match config.image_size
            images = F.interpolate(images, size=(self.config.image_size, self.config.image_size))

            dataset = TensorDataset(images)
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

            # Debug: print first batch shape
            for batch in dataloader:
                print("Batch shape:", batch[0].shape)
                break

            return dataloader

        except Exception as e:
            logging.error("Error loading image data")
            raise CustomException(e, sys)

    # ----------------------
    # Training Loop
    # ----------------------
    def train(self, checkpoint_path=None):
        try:
            dataloader = self.load_data()
            start_epoch = 0

            # Resume from checkpoint if available
            if checkpoint_path:
                start_epoch = self.checkpoint_manager.load_checkpoint(
                    checkpoint_path,
                    self.generator,
                    self.discriminator,
                    self.optimizer_G,
                    self.optimizer_D,
                    self.device
                )
                logging.info(f"Resuming training from epoch {start_epoch}")

            for epoch in range(start_epoch, self.config.epochs):
                for batch_idx, batch in enumerate(dataloader):
                    # Unpack batch correctly
                    real_imgs = batch[0].to(self.device)
                    batch_size = real_imgs.size(0)

                    # Labels for real and fake images
                    real_labels = torch.ones(batch_size, 1).to(self.device)
                    fake_labels = torch.zeros(batch_size, 1).to(self.device)

                    # ----------------------
                    # Train Discriminator
                    # ----------------------
                    noise = torch.randn(batch_size, self.config.latent_dim, 1, 1).to(self.device)
                    fake_imgs = self.generator(noise)

                    # Compute loss
                    real_loss = self.loss_fn(self.discriminator(real_imgs), real_labels)
                    fake_loss = self.loss_fn(self.discriminator(fake_imgs.detach()), fake_labels)
                    d_loss = real_loss + fake_loss

                    # Backprop for Discriminator
                    self.optimizer_D.zero_grad()
                    d_loss.backward()
                    self.optimizer_D.step()

                    # ----------------------
                    # Train Generator
                    # ----------------------
                    noise = torch.randn(batch_size, self.config.latent_dim, 1, 1).to(self.device)
                    gen_imgs = self.generator(noise)
                    g_loss = self.loss_fn(self.discriminator(gen_imgs), real_labels)

                    # Backprop for Generator
                    self.optimizer_G.zero_grad()
                    g_loss.backward()
                    self.optimizer_G.step()

                logging.info(
                    f"Epoch [{epoch+1}/{self.config.epochs}] "
                    f"D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}"
                )

                # Save checkpoint every epoch
                self.checkpoint_manager.save_checkpoint(
                    epoch + 1,
                    self.generator,
                    self.discriminator,
                    self.optimizer_G,
                    self.optimizer_D
                )

            # Save final models & optimizer states
            self.save_models()

        except Exception as e:
            logging.error("Error during GAN training")
            raise CustomException(e, sys)

    # ----------------------
    # Save Models
    # ----------------------
    def save_models(self):
        try:
            os.makedirs(self.config.model_dir, exist_ok=True)

            gen_path = os.path.join(self.config.model_dir, "generator.pth")
            disc_path = os.path.join(self.config.model_dir, "discriminator.pth")

            torch.save(self.generator.state_dict(), gen_path)
            torch.save(self.discriminator.state_dict(), disc_path)

            # Save optimizers
            save_object(os.path.join(self.config.model_dir, "optimizer_G.pkl"), self.optimizer_G.state_dict())
            save_object(os.path.join(self.config.model_dir, "optimizer_D.pkl"), self.optimizer_D.state_dict())

            logging.info(f"Models and optimizer states saved at {self.config.model_dir}")

        except Exception as e:
            logging.error("Error saving models")
            raise CustomException(e, sys)