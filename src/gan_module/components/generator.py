import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=1, feature_maps=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input: [latent_dim, 1, 1]
            nn.ConvTranspose2d(latent_dim, feature_maps*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps*8, feature_maps*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps*4, feature_maps*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps*2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output: [channels, 64, 64]
        )

    def forward(self, z):
        return self.model(z)