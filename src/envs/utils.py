import time
import csv
import json
import numpy as np
import os.path as osp
from gym.core import Wrapper


class Monitor(Wrapper):
    EXT = "monitor.csv"
    f = None

    def __init__(self, env, filename, allow_early_resets=False,
                 reset_keywords=(), info_keywords=()):
        Wrapper.__init__(self, env=env)

        self.tstart = time.time()
        if filename:
            self.results_writer = ResultsWriter(
                filename,
                header={
                    't_start': time.time(),
                    'env_id': env.spec and env.spec.id},
                extra_keys=reset_keywords + info_keywords)
        else:
            self.results_writer = None

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {}

    def reset(self, **kwargs):
        self.reset_state()
        for k in self.reset_keywords:
            v = kwargs.get(k)
            if v is None:
                raise ValueError(f'Expected you to pass kwarg {k} into reset')
            self.current_reset_info[k] = v

        return self.env.reset(**kwargs)

    def reset_state(self):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. "
                "If you want to allow early resets, wrap your env with "
                "Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        self.update(ob, rew, done, info)
        return (ob, rew, done, info)

    def update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {
                "r": round(eprew, 6),
                "l": eplen,
                "t": round(time.time() - self.tstart, 6)}
            for k in self.info_keywords:
                epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(epinfo)
            assert isinstance(info, dict)
            if isinstance(info, dict):
                info['episode'] = epinfo

        self.total_steps += 1

    def close(self):
        super(Monitor, self).close()
        if self.f is not None:
            self.f.close()

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times


class ResultsWriter(object):
    def __init__(self, filename, header='', extra_keys=()):
        self.extra_keys = extra_keys
        assert filename is not None
        if not filename.endswith(Monitor.EXT):
            if osp.isdir(filename):
                filename = osp.join(filename, Monitor.EXT)
            else:
                filename = filename + "." + Monitor.EXT
        self.f = open(filename, "wt")
        if isinstance(header, dict):
            header = '# {} \n'.format(json.dumps(header))
        self.f.write(header)
        self.logger = csv.DictWriter(
            self.f, fieldnames=('r', 'l', 't') + tuple(extra_keys))
        self.logger.writeheader()
        self.f.flush()

    def write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()


def _update(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count


class RunningMeanStd(object):

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.mean, self.var, self.count = _update(
            self.mean, self.var, self.count,
            batch_mean, batch_var, batch_count)


def tile_images(nhwc):
    nhwc = np.asarray(nhwc)
    N, h, w, c = nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    nhwc = np.array(list(nhwc) + [nhwc[0] * 0 for _ in range(N, H*W)])
    HWhwc = np.reshape(nhwc, (H, W, h, w, c))
    HhWwc = HWhwc.transpose(0, 2, 1, 3, 4)
    Hh_Ww_c = np.reshape(HhWwc, (H*h, W*w, c))
    return Hh_Ww_c
