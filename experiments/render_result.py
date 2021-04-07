import numpy as np
from gym import Env
from neat.nn import FeedForwardNetwork


def render_result(
    environment: Env,
    network: FeedForwardNetwork,
    steps: int = 500,
):

    observation = environment.reset()
    for _ in range(steps):
        outputs = network.activate(observation)
        action = np.argmax(outputs)
        observation, *_ = environment.step(action)
        environment.render()
