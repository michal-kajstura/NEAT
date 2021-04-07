import pickle
from pathlib import Path

import gym.wrappers
import neat
from neat import StatisticsReporter, StdOutReporter
from neat.nn import FeedForwardNetwork

from experiments.render_result import render_result
from neat_improved import CONFIGS_PATH
from neat_improved.evaluator import MultipleRunGymEvaluator
from neat_improved.runner import NEATRunner

CONFIG_PATH = CONFIGS_PATH / 'config-cart-pole-v0'
ENV_NAME = 'CartPole-v1'
NUM_GENERATIONS = 100
RENDER_AFTER_TRAINING = True
SAVE_PATH = Path('./best_genome.pkl')


config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    str(CONFIG_PATH),
)

environment = gym.make(ENV_NAME)
evaluator = MultipleRunGymEvaluator(
    environment=environment,
    render=False,
)

runner = NEATRunner(
    config=config,
    evaluator=evaluator,
    reporters=[
        StatisticsReporter(),
        StdOutReporter(
            show_species_detail=False,
        ),
    ]
)

best_genome = runner.run(NUM_GENERATIONS)

with SAVE_PATH.open('wb') as file:
    pickle.dump(best_genome, file)

if RENDER_AFTER_TRAINING:
    network = FeedForwardNetwork.create(best_genome, config)
    render_result(
        environment=environment,
        network=network,
        steps=1000,
    )
