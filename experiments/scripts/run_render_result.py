import pickle
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import neat
from matplotlib import animation
from neat.nn import FeedForwardNetwork

from experiments.utils import render_result
from neat_improved import CONFIGS_PATH

ENV_NAME = 'BipedalWalker-v3'
LOAD_PATH = Path(f'./{ENV_NAME}.pkl')
CONFIG_PATH = CONFIGS_PATH / 'config-bipedal-walker-v3'


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

frames = render_result(
    environment=gym.make(ENV_NAME),
    network=network,
    steps=2000,
)



def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    plt.tight_layout()
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


save_frames_as_gif(frames)
