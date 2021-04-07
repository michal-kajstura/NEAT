import pickle
from pathlib import Path

import gym
import neat
from neat.nn import FeedForwardNetwork

from experiments.render_result import render_result
from neat_improved import CONFIGS_PATH

LOAD_PATH = Path('./best_genome.pkl')
CONFIG_PATH = CONFIGS_PATH / 'config-cart-pole-v0'
ENV_NAME = 'CartPole-v1'
SAVE_PATH = Path('./best_genome.pkl')


config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    str(CONFIG_PATH),
)


with LOAD_PATH.open('rb') as file:
    genome = pickle.load(file)

network = FeedForwardNetwork.create(genome, config)

render_result(
    environment=gym.make(ENV_NAME),
    network=network,
    steps=1000,
)
