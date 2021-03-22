import torch
import torch.nn as nn
from functools import partial
from .utils import init_fn, Flatten, ConvSequence


class ImpalaCNNBody(nn.Module):

    def __init__(self, num_channels, num_initial_blocks=1, depths=(16, 32)):
        super().__init__()
        assert 1 <= num_initial_blocks <= len(depths)

        self._hidden_size = 256

        self.feature_dim = depths[num_initial_blocks-1] * \
            (64 // 2 ** num_initial_blocks) ** 2

        in_channels = num_channels
        nets = []
        for out_channels in depths:
            nets.append(ConvSequence(in_channels, out_channels))
            in_channels = out_channels

        current_dim = depths[-1] * (64 // 2 ** len(depths)) ** 2
        nets.append(
            nn.Sequential(
                nn.LeakyReLU(0.2),
                Flatten(),
                nn.Linear(current_dim, self._hidden_size),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        self.initial_net = nn.Sequential(
            *[nets.pop(0) for _ in range(num_initial_blocks)]
        )
        self.net = nn.Sequential(*nets)

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, states):
        assert torch.float32 == states.dtype
        features = self.initial_net(states)
        return self.net(features)


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        self.linear = nn.Linear(num_inputs, num_outputs).apply(
            partial(init_fn, gain=0.01))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)
