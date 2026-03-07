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

from src.gan_module.components.generator import Generator
from src.gan_module.components.discriminator import Discriminator
from src.gan_module.components.checkpoint_manager import CheckpointManager

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


BASE_DIR = Path(__file__).resolve().parents[3]


@dataclass
class GANTrainerConfig:

    processed_data_path: str = str(BASE_DIR / "artifacts" / "gan" / "processed_images.npy")

    model_dir: str = str(BASE_DIR / "artifacts" / "gan" / "models")

    epochs: int = 50
    batch_size: int = 64
    lr: float = 0.0002
    latent_dim: int = 100


class GANTrainer:

    def __init__(self):

        self.config = GANTrainerConfig()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator().to(self.device)
        logging.info("Generator model initialized")

        self.discriminator = Discriminator().to(self.device)
        logging.info("Discriminator model initialized")

        self.loss_fn = nn.BCELoss()

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.config.lr)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.config.lr)

        self.checkpoint_manager = CheckpointManager("artifacts/gan/checkpoints")

    def load_data(self):

        logging.info("Loading processed image data")

        images = np.load(self.config.processed_data_path, mmap_mode="r")

        images = torch.tensor(images, dtype=torch.float32)
        images = images.unsqueeze(1)

        dataset = TensorDataset(images)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        return dataloader

    def train(self, checkpoint_path=None):

        try:

            dataloader = self.load_data()
            start_epoch = 0

            # Load checkpoint if available
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

                for batch_idx, (real_imgs,) in enumerate(dataloader):

                    real_imgs = real_imgs.to(self.device)
                    batch_size = real_imgs.size(0)

                    real_labels = torch.ones(batch_size, 1).to(self.device)
                    fake_labels = torch.zeros(batch_size, 1).to(self.device)

                    # Train Discriminator
                    noise = torch.randn(batch_size, self.config.latent_dim).to(self.device)

                    fake_imgs = self.generator(noise)

                    real_loss = self.loss_fn(self.discriminator(real_imgs), real_labels)
                    fake_loss = self.loss_fn(self.discriminator(fake_imgs.detach()), fake_labels)

                    d_loss = real_loss + fake_loss

                    self.optimizer_D.zero_grad()
                    d_loss.backward()
                    self.optimizer_D.step()

                    # Train Generator
                    gen_imgs = self.generator(noise)

                    g_loss = self.loss_fn(self.discriminator(gen_imgs), real_labels)

                    self.optimizer_G.zero_grad()
                    g_loss.backward()
                    self.optimizer_G.step()

                logging.info(
                    f"Epoch [{epoch+1}/{self.config.epochs}] "
                    f"D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}"
                )

                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(
                    epoch + 1,
                    self.generator,
                    self.discriminator,
                    self.optimizer_G,
                    self.optimizer_D
                )

            self.save_models()

        except Exception as e:
            raise CustomException(e, sys)

    def save_models(self):

        os.makedirs(self.config.model_dir, exist_ok=True)

        gen_path = os.path.join(self.config.model_dir, "generator.pth")
        disc_path = os.path.join(self.config.model_dir, "discriminator.pth")

        torch.save(self.generator.state_dict(), gen_path)
        torch.save(self.discriminator.state_dict(), disc_path)

        logging.info(f"Models saved at {self.config.model_dir}")