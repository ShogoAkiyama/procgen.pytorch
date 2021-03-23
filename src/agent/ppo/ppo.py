import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from .network import PPONetwork
from .storage import RolloutStorage


class PPO:
    def __init__(self, envs, eval_envs, device, log_dir, num_processes, eval_steps=10**5,
                 num_steps=10**6, batch_size=256, rollout_length=128, lr=2.5e-4,
                 adam_eps=1e-5, gamma=0.99, clip_param=0.1,
                 num_gradient_steps=4, value_loss_coef=0.5, entropy_coef=0.01,
                 lambd=0.95, max_grad_norm=0.5):

        self.envs = envs
        self.eval_envs = eval_envs
        self.device = device

        # PPO parameter
        self.num_processes = num_processes
        self.eval_steps = eval_steps
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.rollout_length = rollout_length

        self.gamma = gamma
        self.clip_param = clip_param
        self.num_gradient_steps = num_gradient_steps
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.lambd = lambd
        self.max_grad_norm = max_grad_norm

        self.num_updates = self.num_steps // (self.num_processes * self.rollout_length)

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
            num_gradient_steps, gamma, lambd,
            envs.observation_space.shape, device)

        # Network
        self.network = PPONetwork(
            envs.observation_space.shape,
            envs.action_space).to(self.device)

        self.optimizer = Adam(
            self.network.parameters(), lr=lr, eps=adam_eps)

        self.episode_rewards = deque(maxlen=10)

    def run(self):
        states = self.envs.reset()
        self.storage.init_states(states)
        episodes = np.zeros(self.num_processes)

        for step in range(self.num_updates):
            for _ in range(self.rollout_length):
                with torch.no_grad():
                    values, actions, action_log_probs = self.network(states)

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

            self.end_rollout(next_states)
            self.update()
            self.evaluate(step)
            self.log(step)

    def end_rollout(self, next_states):
        with torch.no_grad():
            next_values = self.network.get_value(next_states)

        self.storage.end_rolout(next_values)

    def update(self):
        for sample in self.storage.iterate():
            states, actions, pred_values, \
                target_values, log_probs_old, advs = sample

            values, action_log_probs, dist_entropy = \
                self.network.evaluate_action(states, actions)

            ratio = torch.exp(action_log_probs - log_probs_old)

            loss_policy = -torch.min(ratio * advs, torch.clamp(
                ratio, 1.0-self.clip_param, 1.0+self.clip_param) * advs
            ).mean()

            value_pred_clipped = pred_values + (
                values - pred_values
            ).clamp(-self.clip_param, self.clip_param)

            loss_value = 0.5 * torch.max(
                (values - target_values).pow(2),
                (value_pred_clipped - target_values).pow(2)
            ).mean()

            self.optimizer.zero_grad()
            loss = (
                loss_policy
                + self.value_loss_coef * loss_value
                - self.entropy_coef * dist_entropy
            )

            loss.backward()
            nn.utils.clip_grad_norm_(
                self.network.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

    def evaluate(self, step):
        if step % self.eval_steps == 0:
            eval_episodes = 10
            avg_reward = 0
            avg_v = 0
            for _ in range(eval_episodes):
                state, done = self.eval_envs.reset(), False
                while not done:
                    with torch.no_grad():
                        value, action, _ = self.network(state)

                    state, reward, done, _ = self.eval_envs.step(action)
                    done = done.item() == 1

                    avg_reward += reward.item()
                    avg_v += value.item()

            avg_reward /= eval_episodes
            avg_v /= eval_episodes

            print("---------------------------------------")
            print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")

            total_steps = (step + 1) * self.num_processes * self.rollout_length

            self.writer.add_scalar(
                'eval/return', avg_reward, total_steps)
            self.writer.add_scalar(
                'eval/Estimate V', avg_v, total_steps)

    def log(self, step):
        if len(self.episode_rewards) > 1:
            total_steps = (step + 1) * self.num_processes * self.rollout_length
            print(f"\rSteps: {total_steps}   "
                  f"Updates: {step}   "
                  f"Mean Return: {np.mean(self.episode_rewards)}",
                  end='')
            self.writer.add_scalar(
                'return/train', np.mean(self.episode_rewards),
                total_steps)

    def save_model(self, filename):
        torch.save(
            self.network.state_dict(), os.path.join(self.model_dir, filename))
