import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage:

    def __init__(self, rollout_length, num_processes, batch_size,
                 epochs, gamma, lambd, state_shape, device):
        self.states = torch.zeros(
            rollout_length+1, num_processes, *state_shape, device=device)
        self.actions = torch.zeros(
            rollout_length, num_processes, 1, device=device, dtype=torch.long)
        self.rewards = torch.zeros(
            rollout_length, num_processes, 1, device=device)
        self.dones = torch.ones(
            rollout_length+1, num_processes, 1, device=device)

        self.action_log_probs = torch.zeros(
            rollout_length, num_processes, 1, device=device)
        self.pred_values = torch.zeros(
            rollout_length+1, num_processes, 1, device=device)
        self.target_values = torch.zeros(
            rollout_length, num_processes, 1, device=device)

        self.step = 0
        self.rollout_length = rollout_length
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.lambd = lambd
        self.state_shape = state_shape
        self._is_ready = False

    def init_states(self, states):
        self.states[0].copy_(states)

    def insert(self, next_states, actions, rewards, dones,
               action_log_probs, pred_values):
        self.states[self.step+1].copy_(next_states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step+1].copy_(dones)

        self.action_log_probs[self.step].copy_(action_log_probs)
        self.pred_values[self.step].copy_(pred_values)

        self.step = (self.step + 1) % self.rollout_length

    def end_rolout(self, next_value):
        assert not self._is_ready
        self._is_ready = True

        self.pred_values[-1].copy_(next_value)
        adv = 0
        for step in reversed(range(self.rollout_length)):
            # delta_t = r_t + gamma*V_{s_t+1} - V_t
            td_error = self.rewards[step] \
                       + self.gamma * self.pred_values[step+1] * (1-self.dones[step]) \
                       - self.pred_values[step]
            # A = delta_t + \sum_t{gamma*lambd*delta_t}
            adv = td_error + self.gamma * self.lambd * (1-self.dones[step]) * adv

            # V = A - Q
            self.target_values[step] = adv + self.pred_values[step]

    def iterate(self):
        assert self._is_ready

        # Calculate advantages.
        all_advs = self.target_values - self.pred_values[:-1]
        all_advs = (all_advs - all_advs.mean()) / (all_advs.std() + 1e-5)

        for _ in range(self.epochs):
            # Sampler for indices.
            sampler = BatchSampler(
                SubsetRandomSampler(
                    range(self.num_processes * self.rollout_length)),
                self.batch_size, drop_last=True)

            for indices in sampler:
                states = self.states[:-1].view(-1, *self.state_shape)[indices]
                actions = self.actions.view(-1, self.actions.size(-1))[indices]
                pred_values = self.pred_values[:-1].view(-1, 1)[indices]
                target_values = self.target_values.view(-1, 1)[indices]
                action_log_probs = self.action_log_probs.view(-1, 1)[indices]
                advs = all_advs.view(-1, 1)[indices]

                yield states, actions, pred_values, \
                    target_values, action_log_probs, advs

        self.states[0].copy_(self.states[-1])
        self.dones[0].copy_(self.dones[-1])
        self._is_ready = False


class Memory:

    def __init__(self, memory_size, batch_size, epochs, state_shape, action_shape, device):
        self.memory_size = memory_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.state_shape = state_shape

        self.reset()

    def reset(self):
        self.states = torch.zeros((self.memory_size, *self.state_shape))
        self.target_values = torch.zeros((self.memory_size, 1))
        self.pi_old = torch.zeros((self.memory_size, 1))
        self.pred_values = torch.zeros((self.memory_size, 1))

        self.p = 0
        self.q = 0

    def append(self, states, target_values, pred_values):
        if states.device.type == 'cuda':
            states = states.cpu()
            target_values = target_values.cpu()
            pred_values = pred_values.cpu()
        # for i in range((len(states))):
        self.states[self.p:self.p+len(states)] = states
        self.target_values[self.p:self.p+len(target_values)] = target_values
        self.pred_values[self.p:self.p+len(pred_values)] = pred_values
        self.p += len(states)
        # print(self.p, '/', self.memory_size)

    def append_pi_old(self, pi_old):
        if pi_old.device.type == 'cuda':
            pi_old = pi_old.cpu()
        self.pi_old[self.q:self.q+len(pi_old)] = pi_old
        self.q += len(pi_old)

    def iterate(self):
        for _ in range(self.epochs):
            # Sampler for indices.
            sampler = BatchSampler(
                SubsetRandomSampler(range(len(self.states))), self.batch_size, drop_last=True)

            for indices in sampler:
                states = self.states[indices].to(self.device)
                target_values = self.target_values[indices].to(self.device)
                pi_old = self.pi_old[indices].to(self.device)

                pred_values = self.pred_values[indices].to(self.device)

                yield states, target_values, pi_old, pred_values
                # yield states, target_values, pred_values