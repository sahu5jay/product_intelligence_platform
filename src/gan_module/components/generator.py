# src/gan_module/components/generator.py

import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path

from src.shared_utils.logger import logging


@dataclass
class GeneratorConfig:
    latent_dim: int = 100
    img_channels: int = 1
    feature_map_size: int = 64


class Generator(nn.Module):

    def __init__(self, config: GeneratorConfig = GeneratorConfig()):
        super(Generator, self).__init__()

        self.config = config

        self.model = nn.Sequential(

            # Input: noise vector (latent_dim)
            nn.Linear(self.config.latent_dim, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),

            # Output layer
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

        logging.info("Generator model initialized")

    def forward(self, z):

        img = self.model(z)

        img = img.view(img.size(0), 1, 28, 28)

        return img