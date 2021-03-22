import time
import torch
import numpy as np
from collections import deque
from gym.spaces.box import Box

from .abstract import (
    VecEnvObservationWrapper,
    VecEnvWrapper
)
from .utils import (
    ResultsWriter,
    RunningMeanStd
)


class VecExtractDictObs(VecEnvObservationWrapper):
    def __init__(self, venv, key):
        self.key = key
        super().__init__(
            venv=venv,
            observation_space=venv.observation_space.spaces[self.key])

    def process(self, obs):
        return obs[self.key]


class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, filename=None, keep_buf=0, info_keywords=()):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()
        if filename:
            self.results_writer = ResultsWriter(
                filename, header={'t_start': self.tstart},
                extra_keys=info_keywords)
        else:
            self.results_writer = None
        self.info_keywords = info_keywords
        self.keep_buf = keep_buf
        if self.keep_buf:
            self.epret_buf = deque([], maxlen=keep_buf)
            self.eplen_buf = deque([], maxlen=keep_buf)

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {
                    'r': ret,
                    'l': eplen,
                    't': round(time.time() - self.tstart, 6)}
                for k in self.info_keywords:
                    epinfo[k] = info[k]
                info['episode'] = epinfo
                if self.keep_buf:
                    self.epret_buf.append(ret)
                    self.eplen_buf.append(eplen)
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(epinfo)
                newinfos[i] = info
        return obs, rews, dones, newinfos


class VecNormalize(VecEnvWrapper):

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10.,
                 gamma=0.99, epsilon=1e-8, use_tf=False):
        VecEnvWrapper.__init__(self, venv)

        self.ob_rms = RunningMeanStd(
            shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(
                rews / np.sqrt(self.ret_rms.var + self.epsilon),
                -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip(
                (obs - self.ob_rms.mean) /
                np.sqrt(self.ob_rms.var + self.epsilon),
                -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)


class VecPyTorchProcgen(VecEnvWrapper):
    def __init__(self, venv, device):

        super(VecPyTorchProcgen, self).__init__(venv)

        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [3, 64, 64],
            dtype=self.observation_space.dtype)

        self.device = device

    def reset(self):
        obs = self.venv.reset()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs.copy()).float().to(self.device) / 255.
        # obs = torch.FloatTensor(obs / 255., device=self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor) or len(actions.shape) > 1:
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs.copy()).float().to(self.device) / 255.
        # obs = torch.FloatTensor(obs / 255., device=self.device)
        rewards = torch.from_numpy(rewards).unsqueeze(dim=1).float()
        dones = torch.from_numpy(dones).unsqueeze(dim=1).float()
        return obs, rewards, dones, infos
