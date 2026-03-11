import torch
import torch.nn as nn
import sys

from torch.nn.utils import spectral_norm

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


class Discriminator(nn.Module):
    """
    Optimized DCGAN Discriminator
    Classifies real vs fake images
    """

    def __init__(self, channels=1, feature_maps=64):

        try:

            super().__init__()

            logging.info("Initializing Discriminator model")

            self.net = nn.Sequential(

                spectral_norm(
                    nn.Conv2d(channels, feature_maps, 4, 2, 1)
                ),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(
                    nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1)
                ),
                nn.BatchNorm2d(feature_maps * 2),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(
                    nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1)
                ),
                nn.BatchNorm2d(feature_maps * 4),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(
                    nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1)
                ),
                nn.BatchNorm2d(feature_maps * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(feature_maps * 8, 1, 4, 1, 0)

            )

            logging.info("Discriminator initialized successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def forward(self, x):
        try:
            return self.net(x).view(-1)

        except Exception as e:
            logging.error("Error in Discriminator forward pass")
            raise CustomException(e, sys)