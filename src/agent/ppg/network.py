import torch
import torch.nn as nn
from src.model.network import ImpalaCNNBody, Categorical, initialize_xavier


class PolicyNetwork(nn.Module):
    def __init__(self, obs_shape, action_space):
        super().__init__()
        self.base = ImpalaCNNBody(obs_shape[0])
        self.dist = Categorical(self.base.output_size, action_space).apply(initialize_xavier)
        self.v_head = nn.Linear(self.base.output_size, 1).apply(initialize_xavier)

    def forward(self, states, determistic=False):
        actor_features = self.base(states)
        dists = self.dist(actor_features)

        if determistic:
            actions = dists.mode()
        else:
            actions = dists.sample()

        action_log_probs = dists.log_probs(actions)

        values = self.v_head(actor_features)

        return values, actions, action_log_probs

    def get_value(self, states):
        actor_features = self.base(states)
        values = self.v_head(actor_features)
        return values

    def get_aux(self, states):
        actor_features = self.base(states)
        dists = self.dist(actor_features)
        values = self.v_head(actor_features)
        return values, dists.probs

    def evaluate_action(self, states, actions):
        actor_features = self.base(states)
        dists = self.dist(actor_features)

        action_log_probs = dists.log_probs(actions)
        dist_entropy = dists.entropy().mean()

        values = self.v_head(actor_features)

        return values, action_log_probs, dist_entropy


class ValueNetwork(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        self.base = ImpalaCNNBody(obs_shape[0])
        self.v_head = nn.Linear(self.base.output_size, 1).apply(initialize_xavier)

    def forward(self, states):
        actor_features = self.base(states)
        values = self.v_head(actor_features)

        return values

    def get_value(self, states):
        actor_features = self.pi_base(states)
        values = self.pi_v_head(actor_features)
        return values



