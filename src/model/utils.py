import torch.nn as nn


def init_fn(m, gain=1):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        nn.init.constant_(m.bias.data, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ResidualBlock(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_channels, num_channels, 3, 1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, 1, padding=1),
        )

    def forward(self, x):
        return self.net(x).add_(x)


class ConvSequence(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
            nn.MaxPool2d(3, 2, padding=1),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels)
        )

    def forward(self, x):
        return self.net(x)
