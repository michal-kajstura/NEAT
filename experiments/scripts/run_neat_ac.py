import multiprocessing
import random

import gym
import neat
import numpy as np
import torch
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy

from neat_improved import PROJECT_PATH
from neat_improved.neat import NEAT_CONFIGS
from neat_improved.neat_rl.neat_ac import NEATACTrainer

SEED = 2021

STOP_TIME = None
MAX_FRAMES = int(5e5)
N_REPEATS = 3

LOGGING_DIR = PROJECT_PATH.parent / 'logs_neat_rl'
LOGGING_DIR.mkdir(exist_ok=True)

enviroments = (
    'MountainCarContinuous-v0',
    'CartPole-v0',
    'Pendulum-v0',
    'LunarLander-v2',
    'BipedalWalker-v3',
    'BipedalWalkerHardcore-v3',
)

USE_GPU = True
NUM_GENERATIONS = None
NUM_WORKERS = multiprocessing.cpu_count()

np.random.seed(SEED)
torch.random.manual_seed(SEED)
random.seed(SEED)

environment_name = enviroments[0]
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    str(NEAT_CONFIGS[environment_name]),
)
env = gym.make(environment_name)

a2c = A2C(
    MlpPolicy,
    env
)
a2c.learn(100)

neat_ac_trainer = NEATACTrainer(
    env=env,
    config=config,
)

neat_ac_trainer.train(10, None)