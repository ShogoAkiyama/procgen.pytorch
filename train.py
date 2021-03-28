import os
import yaml
import argparse
import torch
from datetime import datetime

from src.utils import fix_seed
from src.envs.envs import make_env
from src.agent import *

AGENT_CLASS = {
    'ppo': (PPO, 'ppo,yaml'),
    'dqn': (DQN, 'dqn,yaml'),
    'ppg': (PPG, 'ppg,yaml'),
}


def train(args):
    with open(os.path.join('config', 'procgen.yaml')) as f:
        procgen_config = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join('config', args.agent+'.yaml')) as f:
        agent_config = yaml.load(f, Loader=yaml.SafeLoader)

    device = torch.device(
        'cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        args.log_dir,
        args.agent,
        args.env_name.split('NoFrameskip')[0],
        args.agent+f'-{args.seed}-{time}')

    # fix seed
    fix_seed(args.seed)

    # make envs
    envs = make_env(args.env_name, args.num_processes, device, **procgen_config)
    eval_envs = make_env(args.env_name, 1, device, **procgen_config)

    # make agent
    agent, param_file = AGENT_CLASS[args.agent]

    # run
    agent(envs, eval_envs, device, log_dir, args.num_processes,
          **agent_config).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--agent', type=str, default='ppo')
    parser.add_argument('--cuda', action='store_true')

    # Procgen Argments
    parser.add_argument('--env_name', type=str, default='bigfish')
    parser.add_argument('--num_processes', type=int, default=64)

    args = parser.parse_args()

    train(args)
