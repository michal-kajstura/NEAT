import multiprocessing
import pickle
from pathlib import Path

import gym
import neat
from neat.nn import FeedForwardNetwork

from experiments.utils import render_result, run
from neat_improved import CONFIGS_PATH

EXPERIMENTS = [
    {
        'env_name': 'LunarLander-v2',
        'config_path': CONFIGS_PATH / 'config-lunar-lander-v2',
    },
    {
        'env_name': 'BipedalWalker-v3',
        'config_path': CONFIGS_PATH / 'config-bipedal-walker-v3',
    },
    {
        'env_name': 'CartPole-v0',
        'config_path': CONFIGS_PATH / 'config-cart-pole-v0',
    },
    {
        'env_name': 'MountainCarContinuous-v0',
        'config_path': CONFIGS_PATH / 'config-mountain-car-continous-v0',
    },
    {
        'env_name': 'Pendulum-v0',
        'config_path': CONFIGS_PATH / 'config-pendulum-v0',
    },
]
NUM_GENERATIONS = 100
RENDER_AFTER_TRAINING = True
SAVE_DIR = Path('.')
NUM_WORKERS = multiprocessing.cpu_count()
NUM_REPEATS = 3
LOGGING_DIR = Path('./logs1')


for experiment in EXPERIMENTS:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(experiment['config_path']),
    )

    name = experiment['env_name']
    environment = gym.make(name)

    best_genome = run(
        environment,
        config,
        NUM_GENERATIONS,
        NUM_WORKERS,
        logging_root=LOGGING_DIR,
        num_repeats=NUM_REPEATS,
    )

    with (SAVE_DIR / (name + '.pkl')).open('wb') as file:
        pickle.dump(best_genome, file)

    if RENDER_AFTER_TRAINING:
        network = FeedForwardNetwork.create(best_genome, config)
        render_result(
            environment=environment,
            network=network,
            steps=1000,
        )
