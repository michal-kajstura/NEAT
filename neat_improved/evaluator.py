import abc
from typing import Optional

import numpy as np
from gym import Env
from neat import Config, DefaultGenome
from neat.nn import FeedForwardNetwork


class GymEvaluator(abc.ABC):
    def __init__(
        self,
        environment: Env,
        render: bool = False,
    ):
        self._environment = environment
        self._render = render

    @abc.abstractmethod
    def evaluate(
        self,
        genome: DefaultGenome,
        config: Config,
    ) -> float:
        pass


class MultipleRunGymEvaluator(GymEvaluator):
    def __init__(
        self,
        environment: Env,
        runs_per_network: int = 1,
        max_steps: Optional[int] = None,
        render: bool = False,
    ):
        super().__init__(
            environment=environment,
            render=render
        )
        self._runs_per_network = runs_per_network
        self._max_steps = max_steps or float('inf')

    def evaluate(
        self,
        genome: DefaultGenome,
        config: Config
    ) -> float:
        network = FeedForwardNetwork.create(genome, config)
        fitness_scores = [self._run_episode(network) for _ in range(self._runs_per_network)]
        return np.mean(fitness_scores)

    def _run_episode(
        self,
        network: FeedForwardNetwork,
    ) -> float:
        observation = self._environment.reset()

        fitness = 0.0
        done = False
        step = 0
        while not done:
            if step > self._max_steps:
                break

            output = network.activate(observation)
            action = np.argmax(output)
            observation, reward, done, _ = self._environment.step(action)
            fitness += reward
            step += 1

        return fitness
