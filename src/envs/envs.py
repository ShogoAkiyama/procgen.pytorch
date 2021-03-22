from procgen import ProcgenEnv
from src.envs.wrappers import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize,
    VecPyTorchProcgen
)


def make_env(env_name, num_processes, device,
             num_levels, start_level, distribution_mode):
    print('make_env')
    venv = ProcgenEnv(
        env_name=env_name,
        num_envs=num_processes,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode
    )
    venv = VecExtractDictObs(venv, 'rgb')
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    venv = VecPyTorchProcgen(venv, device)

    return venv
