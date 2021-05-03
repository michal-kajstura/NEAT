import multiprocessing
import random
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

from experiments.utils import run_neat, run_actor_critic
from neat_improved import PROJECT_PATH

RUN_ACTOR_CRITIC = True
RUN_NEAT = True
SEED = 2021

STOP_TIME = None
MAX_FRAMES = int(5e6)
N_REPEATS = 3

LOGGING_DIR = PROJECT_PATH.parent / 'logs_neat_vs_rl'
LOGGING_DIR.mkdir(exist_ok=True)

enviroments = (
    'LunarLander-v2',
    'MountainCarContinuous-v0',
    'BipedalWalker-v3',
    'BipedalWalkerHardcore-v3',
    'Pendulum-v0',
    'CartPole-v0',
)

USE_GPU = True
NUM_GENERATIONS = None
NUM_WORKERS = multiprocessing.cpu_count()


@dataclass
class NEATExperimentConfig:
    runs_per_network: int = 1
    name: str = 'neat'


@dataclass
class A2CExperimentConfig:
    lr: float = 7e-4
    gamma: float = 0.99
    normalize_advantage: bool = False
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01,
    common_stem: bool = False
    name: str = 'ac'


def run_experiment(
    env_name: str,
    experiment: Union[NEATExperimentConfig, A2CExperimentConfig],
    seed: int,
):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

    if isinstance(experiment, NEATExperimentConfig):
        run_neat(
            env_name,
            max_frames=MAX_FRAMES,
            stop_time=STOP_TIME,
            num_workers=NUM_WORKERS,
            logging_dir=LOGGING_DIR / experiment.name,
            runs_per_network=experiment.runs_per_network,
            seed=seed,
        )
    elif isinstance(experiment, A2CExperimentConfig):
        run_actor_critic(
            environment_name=env_name,
            max_frames=MAX_FRAMES,
            stop_time=STOP_TIME,
            logging_dir=LOGGING_DIR / experiment.name,
            use_gpu=USE_GPU,
            lr=experiment.lr,
            gamma=experiment.gamma,
            normalize_advantage=experiment.normalize_advantage,
            value_loss_coef=experiment.value_loss_coef,
            seed=seed,
        )


experiment_configs = (
    NEATExperimentConfig(name='neat'),
    A2CExperimentConfig(name='a2c'),
    A2CExperimentConfig(common_stem=True, name='ac_common_stem'),
    # A2CExperimentConfig(common_stem=True, value_loss_coef=0.1),
    # A2CExperimentConfig(normalize_advantage=False),
    # A2CExperimentConfig(lr=7e-3),
    # A2CExperimentConfig(entropy_coef=0.1),
    # A2CExperimentConfig(entropy_coef=0.001),
)

if __name__ == '__main__':
    for config in experiment_configs:
        for env_name in enviroments:
            for repeat in range(N_REPEATS):
                run_experiment(env_name, config, seed=SEED + repeat)
