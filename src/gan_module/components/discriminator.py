# src/gan_module/components/discriminator.py

import torch
import torch.nn as nn
from dataclasses import dataclass

from src.shared_utils.logger import logging


@dataclass
class DiscriminatorConfig:
    img_channels: int = 1
    img_size: int = 28


class Discriminator(nn.Module):

    def __init__(self, config: DiscriminatorConfig = DiscriminatorConfig()):
        super(Discriminator, self).__init__()

        self.config = config
        self.img_shape = (config.img_channels, config.img_size, config.img_size)

        self.model = nn.Sequential(

            # Flatten image
            nn.Flatten(),

            nn.Linear(int(torch.prod(torch.tensor(self.img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer (Real or Fake)
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        logging.info("Discriminator model initialized")

    def forward(self, img):

        validity = self.model(img)

        return validity