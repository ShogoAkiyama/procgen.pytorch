import torch.nn as nn
from src.model.network import ImpalaCNNBody


class DQNNetwork(nn.Module):

    def __init__(self, obs_shape, action_space):
        super().__init__()

        self.base = ImpalaCNNBody(obs_shape[0])
        self.q_net = nn.Sequential(
            nn.Linear(self.base.output_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_space.n),
        )

    def forward(self, states):
        return self.q_net(self.base(states))

    def calculate_q(self, states):
        return self.q_net(self.base(states))
