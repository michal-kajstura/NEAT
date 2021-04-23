import multiprocessing

import gym
import neat

from experiments.utils import run_neat, run_actor_critic
from neat_improved import NEAT_CONFIGS_PATH, PROJECT_PATH

RUN_ACTOR_CRITIC = True
RUN_NEAT = True

STOP_TIME = 1 * 60

LOGGING_DIR = PROJECT_PATH / 'logs'
LOGGING_DIR.mkdir(exist_ok=True)

experiments = (
    'CartPole-v0',
    'LunarLander-v2',
)

# Actor Critic stuff
MAX_EPISODES = None
LR = 0.0001
GAMMA = 0.99
USE_GPU = True  # probably should be set to False
ACTOR_CRITIC_LOGGING_DIR = LOGGING_DIR / 'actor_critic'
ACTOR_CRITIC_LOGGING_DIR.mkdir(exist_ok=True)


# NEAT stuff
NEAT_CONFIGS = {
    'LunarLander-v2': NEAT_CONFIGS_PATH / 'config-lunar-lander-v2',
    'BipedalWalker-v3': NEAT_CONFIGS_PATH / 'config-bipedal-walker-v3',
    'CartPole-v0': NEAT_CONFIGS_PATH / 'config-cart-pole-v0',
    'MountainCarContinuous-v0': NEAT_CONFIGS_PATH / 'config-mountain-car-continous-v0',
    'Pendulum-v0': NEAT_CONFIGS_PATH / 'config-pendulum-v0',
}
NUM_GENERATIONS = None
NUM_WORKERS = multiprocessing.cpu_count()  # probably should be set to 1
NUM_REPEATS = 1
NEAT_LOGGING_DIR = LOGGING_DIR / 'neat'
NEAT_LOGGING_DIR.mkdir(exist_ok=True)


for env_name in experiments:
    environment = gym.make(env_name)

    if RUN_NEAT:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(NEAT_CONFIGS[env_name]),
        )

        best_genome = run_neat(
            environment,
            config,
            num_generations=NUM_GENERATIONS,
            stop_time=STOP_TIME,
            num_workers=NUM_WORKERS,
            logging_root=NEAT_LOGGING_DIR,
        )

    if RUN_ACTOR_CRITIC:
        run_actor_critic(
            environment=environment,
            num_iterations=MAX_EPISODES,
            lr=LR,
            gamma=GAMMA,
            stop_time=STOP_TIME,
            use_gpu=USE_GPU,
            logging_root=ACTOR_CRITIC_LOGGING_DIR,
        )
