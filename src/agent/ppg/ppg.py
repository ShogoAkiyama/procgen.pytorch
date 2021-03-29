import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from torch import distributions

from .network import PolicyNetwork, ValueNetwork
from .storage import RolloutStorage, Memory


class PPG:
    def __init__(self, envs, eval_envs, device, log_dir, num_processes, eval_steps=10,
                 num_steps=10**8, batch_size=256, rollout_length=128, lr=5.e-4,
                 adam_eps=1e-5, gamma=0.999, clip_param=0.2, epochs=1, entropy_coef=0.01,
                 lambd=0.95, max_grad_norm=0.5, aux_batch_size=16, aux_steps=32, aux_epochs=6,
                 kl_coef=0.):

        self.envs = envs
        self.eval_envs = eval_envs
        self.device = device
        self.state_shape = envs.observation_space.shape
        self.action_shape = envs.action_space.n

        # PPO parameter
        self.num_processes = num_processes
        self.eval_steps = eval_steps
        self.num_steps = num_steps
        self.rollout_length = rollout_length
        self.batch_size = batch_size

        self.gamma = gamma
        self.clip_param = clip_param
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.lambd = lambd
        self.max_grad_norm = max_grad_norm

        self.num_updates = self.num_steps // (self.num_processes * self.rollout_length)

        # auxiliary parameter
        memory_size = aux_steps * num_processes * rollout_length
        self.aux_steps = aux_steps

        self.memory_size = memory_size
        self.memory = Memory(
            memory_size, aux_batch_size, aux_epochs,
            self.state_shape, self.action_shape, self.device)

        self.model_dir = os.path.join(log_dir, 'model')
        summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        self.writer = SummaryWriter(log_dir=summary_dir)

        # storage
        self.storage = RolloutStorage(
            rollout_length, num_processes, batch_size,
            epochs, gamma, lambd,
            envs.observation_space.shape, device)

        # Network
        self.policy_network = PolicyNetwork(
            envs.observation_space.shape,
            envs.action_space.n).to(self.device)

        self.value_network = ValueNetwork(
            envs.observation_space.shape).to(self.device)

        self.policy_optimizer = Adam(
            self.policy_network.parameters(), lr=lr, eps=adam_eps)

        self.value_optimizer = Adam(
            self.value_network.parameters(), lr=lr, eps=adam_eps)

        self.episode_rewards = deque(maxlen=10)
        self.step = 0
        self.total_step = 0

        self.kl_coef = kl_coef

    def run(self):
        states = self.envs.reset()
        self.storage.init_states(states)
        episodes = np.zeros(self.num_processes)

        for step in range(self.num_updates):
            for _ in range(self.rollout_length):
                with torch.no_grad():
                    _, actions, action_log_probs = self.policy_network(states)
                    values = self.value_network(states)

                next_states, rewards, dones, infos = self.envs.step(actions)

                episodes += rewards.cpu().numpy().flatten()

                for i, done in enumerate(dones.cpu().detach().numpy()):
                    if done:
                        self.episode_rewards.append(episodes[i])
                        episodes[i] = 0

                self.storage.insert(
                    next_states, actions, rewards, dones,
                    action_log_probs, values)

                states = next_states

            self.step = step
            self.total_step = (step + 1) * self.num_processes * self.rollout_length
            self.end_rollout(next_states)
            self.update()
            self.evaluate()
            self.log()

    def end_rollout(self, next_states):
        with torch.no_grad():
            next_values = self.value_network(next_states)

        self.storage.end_rolout(next_values)

    def update(self):
        for sample in self.storage.iterate():
            self.policy_update(sample)
            self.value_update(sample)

            states, _, pred_values, target_values, _, _ = sample

            self.memory.append(states, target_values, pred_values)

        if (self.step+1) % self.aux_steps == 0:
            if self.kl_coef > 0:
                with torch.no_grad():
                    for states in self.memory.state_iterate():
                        _, pi_old = self.policy_network.get_aux(states.to(self.device))
                        self.memory.append_pi_old(pi_old)

            for sample in self.memory.iterate():
                self.aux_update(sample)

            self.memory.reset()

    def aux_update(self, sample):
        states, target_values, pi_old, pred_values = sample

        # policy update
        pi_values, pi_probs = self.policy_network.get_aux(states)
        log_pi_probs = (
            pi_probs + (pi_probs == 0.0).float() * 1e-8
        ).log()
        log_pi_old = (
            pi_old + (pi_old == 0.0).float() * 1e-8
        ).log()

        kl = (pi_old * (log_pi_old - log_pi_probs)).mean()

        value_pred_clipped = pred_values + (
            pi_values - pred_values
        ).clamp(-self.clip_param, self.clip_param)

        aux_loss_values = 0.5 * torch.max(
            (pi_values - target_values).pow(2),
            (value_pred_clipped - target_values).pow(2)
        ).mean()

        loss = (
            aux_loss_values + self.kl_coef * kl
        )

        self.policy_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), self.max_grad_norm
        )
        self.policy_optimizer.step()

        # value update
        values = self.value_network(states)

        value_pred_clipped = pred_values + (
            values - pred_values
        ).clamp(-self.clip_param, self.clip_param)

        loss_value = 0.5 * torch.max(
            (values - target_values).pow(2),
            (value_pred_clipped - target_values).pow(2)
        ).mean()

        self.value_optimizer.zero_grad()
        loss_value.backward()
        nn.utils.clip_grad_norm_(
            self.value_network.parameters(), self.max_grad_norm
        )
        self.value_optimizer.step()

    def policy_update(self, sample):
        states, actions, _, _, log_probs_old, advs = sample

        pi_values, action_log_probs, dist_entropy = \
            self.policy_network.evaluate_action(states, actions)

        ratio = torch.exp(action_log_probs - log_probs_old)

        loss_policy = torch.min(ratio * advs, torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advs
        ).mean()

        loss = - loss_policy - self.entropy_coef * dist_entropy

        self.policy_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), self.max_grad_norm
        )
        self.policy_optimizer.step()

        self.writer.add_scalar(
            'train/loss_policy', loss_policy.item(),
            self.total_step)
        self.writer.add_scalar(
            'train/loss_entropy', dist_entropy.item(),
            self.total_step)

    def value_update(self, sample):
        states, _, pred_values, target_values, _, _ = sample

        values = self.value_network(states)

        # loss_value = 0.5 * (values - target_values).pow(2).mean()

        value_pred_clipped = pred_values + (
            values - pred_values
        ).clamp(-self.clip_param, self.clip_param)

        loss_value = 0.5 * torch.max(
            (values - target_values).pow(2),
            (value_pred_clipped - target_values).pow(2)
        ).mean()

        self.value_optimizer.zero_grad()
        loss_value.backward()
        nn.utils.clip_grad_norm_(
            self.value_network.parameters(), self.max_grad_norm
        )
        self.value_optimizer.step()

        self.writer.add_scalar(
            'train/loss_value', loss_value.item(),
            self.total_step)

    def evaluate(self):
        if self.step % self.eval_steps == 0:
            eval_episodes = 10
            avg_reward = 0
            avg_v = 0
            steps = 0
            for _ in range(eval_episodes):
                state, done = self.eval_envs.reset(), False
                while not done:
                    with torch.no_grad():
                        _, action, _ = self.policy_network(state)
                        value = self.value_network(state)

                    state, reward, done, _ = self.eval_envs.step(action)
                    done = done.item() == 1

                    avg_reward += reward.item()
                    avg_v += value.item()
                    steps += 1

            avg_reward /= eval_episodes
            avg_v /= steps

            # print("---------------------------------------")
            print(f"\nEvaluation over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")

            self.writer.add_scalar(
                'return/eval', avg_reward, self.total_step)
            self.writer.add_scalar(
                'eval/Estimate V', avg_v, self.total_step)

    def log(self):
        if len(self.episode_rewards) > 1:
            print(f"\rSteps: {self.total_step}   "
                  f"Updates: {self.step}   "
                  f"Mean Return: {np.mean(self.episode_rewards)}",
                  end='')
            self.writer.add_scalar(
                'return/train', np.mean(self.episode_rewards),
                self.total_step)

    def save_model(self, filename):
        torch.save(
            self.network.state_dict(), os.path.join(self.model_dir, filename))
