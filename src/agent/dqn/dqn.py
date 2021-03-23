import os
import numpy as np
from collections import deque
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from .network import DQNNetwork
from .utils import LinearAnneaer
from .memory import LazyMultiStepMemory


class DQN:

    def __init__(self, envs, eval_envs, device, log_dir, num_processes,
                 eval_steps=10**5, target_update_steps=10**4, update_steps=4,
                 num_steps=5*(10**7), epsilon_decay_steps=25*(10**4),
                 start_steps=5*(10**4), max_episode_steps=27*(10**3),
                 lr=2.5e-4, adam_eps=1.e-5, epsilon_train=0.01, epsilon_eval=0.001,
                 memory_size=10**6, batch_size=32, gamma=0.99, multi_step=1):

        self.envs = envs
        self.eval_envs = eval_envs
        self.device = device

        # DQN parameter
        self.eval_steps = eval_steps
        self.target_update_steps = target_update_steps
        self.update_steps = update_steps
        self.num_steps = num_steps
        self.start_steps = start_steps
        self.max_episode_steps = max_episode_steps

        self.model_dir = os.path.join(log_dir, 'model')
        summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        self.writer = SummaryWriter(log_dir=summary_dir)

        self.network = DQNNetwork(
            envs.observation_space.shape,
            envs.action_space).to(self.device)
        self.target_network = DQNNetwork(
            envs.observation_space.shape,
            envs.action_space
        ).to(self.device)

        self.target_network.load_state_dict(
            self.network.state_dict())

        self.optimizer = Adam(
            self.network.parameters(), lr=lr, eps=adam_eps)

        self.epsilon_anneal = LinearAnneaer(
            1.0, epsilon_train, epsilon_decay_steps)
        self.epsilon_eval = epsilon_eval

        self.memory = LazyMultiStepMemory(
            memory_size, batch_size, self.envs.observation_space.shape,
            self.device, gamma, multi_step)

        self.double_q = False
        self.steps = 0
        self.best_eval_score = -np.inf
        self.episode_rewards = deque(maxlen=10)
        self.gamma_n = gamma ** multi_step

    def run(self):
        while self.steps < self.num_steps:
            episode_steps = 0
            episode_reward = 0.
            done = False
            state = self.envs.reset()
            while (not done) and episode_steps <= self.max_episode_steps:

                if self.is_random(eval=False):
                    action = self.envs.action_space.sample()
                else:
                    action = self.exploit(state)

                next_state, reward, done, _ = self.envs.step(action)

                self.memory.append(state, action, reward, next_state, done)

                self.steps += 1
                episode_steps += 1
                done = done.item() == 1
                episode_reward += reward
                state = next_state

                self.epsilon_anneal.step()

                if self.steps % self.target_update_steps == 0:
                    self.target_network.load_state_dict(
                        self.network.state_dict())

                if self.steps >= self.start_steps \
                        and self.steps % self.update_steps == 0:
                    self.learn()

                if self.steps % self.eval_steps == 0:
                    self.evaluate()
                    self.save_model('final')
                    self.network.train()

            self.episode_rewards.append(episode_reward)
            self.log()

        self.save_model(filename='final_model.pth')

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample()

        curr_q = self.network(states).gather(1, actions)

        with torch.no_grad():
            if self.double_q:
                next_actions = self.network(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_network(next_states).gather(1, next_actions)
            else:
                q_value = self.target_network(next_states)
                next_q = q_value.gather(1, q_value.argmax(dim=1, keepdim=True))

        target_q = rewards + (1 - dones) * self.gamma_n * next_q

        loss = torch.mean((target_q - curr_q) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def exploit(self, state):
        with torch.no_grad():
            action = self.network.calculate_q(states=state).argmax().item()
        return action

    def is_random(self, eval=False):
        if self.steps < self.start_steps:
            return True
        if eval:
            return np.random.rand() < self.epsilon_eval
        return np.random.rand() < self.epsilon_anneal.get()

    def evaluate(self):
        self.network.eval()
        num_episodes = 0
        num_steps = 0
        total_rewards = 0.0
        eval_episodes = 10

        for _ in range(eval_episodes):
            state = self.eval_envs.reset()
            episode_steps = 0
            episode_rewards = 0.0
            done = False
            while (not done) and episode_steps <= self.max_episode_steps:
                if self.is_random(eval=True):
                    action = self.envs.action_space.sample()
                else:
                    action = self.exploit(state)

                next_state, reward, done, _ = self.eval_envs.step(action)
                num_steps += 1
                episode_steps += 1
                episode_rewards += reward.item()
                done = done.item() == 1
                state = next_state

            num_episodes += 1
            total_rewards += episode_rewards

        avg_reward = total_rewards / num_episodes

        if avg_reward > self.best_eval_score:
            self.best_eval_score = avg_reward
            self.save_model('best')

        print("\n---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")

        # We log evaluation results along with training frames = 4 * steps.
        self.writer.add_scalar(
            'return/test', avg_reward, 4 * self.steps)

    def log(self):
        if len(self.episode_rewards) > 1:
            print(f"\rSteps: {self.steps}   "
                  f"Mean Return: {np.mean(self.episode_rewards):.3f}",
                  end='')
            self.writer.add_scalar(
                'return/train', np.mean(self.episode_rewards),
                4 * self.steps)

    def save_model(self, filename):
        torch.save(
            self.network.state_dict(), os.path.join(self.model_dir, filename))
