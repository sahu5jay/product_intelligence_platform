import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels=1, feature_maps=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),          # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps*2, 4, 2, 1, bias=False),   # 32 -> 16
            nn.BatchNorm2d(feature_maps*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps*2, feature_maps*4, 4, 2, 1, bias=False), # 16 -> 8
            nn.BatchNorm2d(feature_maps*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps*4, feature_maps*8, 4, 2, 1, bias=False), # 8 -> 4
            nn.BatchNorm2d(feature_maps*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps*8, 1, 4, 1, 0, bias=False),               # 4 -> 1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)