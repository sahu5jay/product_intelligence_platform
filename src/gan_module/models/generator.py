import torch
import torch.nn as nn
import sys

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


class Generator(nn.Module):
    """
    Optimized DCGAN Generator
    Converts random noise into 64x64 image
    """

    def __init__(self, latent_dim=100, channels=1, feature_maps=64):
        try:
            super().__init__()

            logging.info("Initializing Generator model")

            self.net = nn.Sequential(

                nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(feature_maps * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps),
                nn.ReLU(True),

                nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),

                nn.Tanh()
            )

            logging.info("Generator initialized successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def forward(self, z):
        try:
            return self.net(z)
        except Exception as e:
            logging.error("Error in Generator forward pass")
            raise CustomException(e, sys)