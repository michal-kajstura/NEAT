import abc
from typing import Optional, Tuple

import gym
from gym import Env
from neat import Config, DefaultGenome
from neat.nn import FeedForwardNetwork

from neat_improved.neat.action_handler import handle_action


class GymEvaluator(abc.ABC):
    def __init__(
        self,
        environment_name: str,
        render: bool = False,
    ):
        self._environment_name = environment_name
        self._render = render

    @property
    @abc.abstractmethod
    def num_frames(self) -> int:
        pass

    @num_frames.setter
    @abc.abstractmethod
    def num_frames(self, i):
        pass

    @abc.abstractmethod
    def evaluate(
        self,
        genome: DefaultGenome,
        config: Config,
    ) -> Tuple[float, int]:
        pass


class MultipleRunGymEvaluator(GymEvaluator):
    def __init__(
        self,
        environment_name: str,
        runs_per_network: int = 1,
        max_steps: Optional[int] = None,
        render: bool = False,
    ):
        super().__init__(
            environment_name=environment_name,
            render=render,
        )
        self._runs_per_network = runs_per_network
        self._max_steps = max_steps or float('inf')
        self._num_frames = 0

    @property
    def num_frames(self):
        return self._num_frames

    @num_frames.setter
    def num_frames(self, i):
        self._num_frames = i

    def evaluate(self, genome: DefaultGenome, config: Config) -> Tuple[float, int]:
        network = FeedForwardNetwork.create(genome, config)
        environment = gym.make(self._environment_name)

        fitness, frames = 0., 0
        for _ in range(self._runs_per_network):
            fit, fr = self._run_episode(network, environment)
            fitness += fit
            frames += fr

        return fitness / self._runs_per_network, frames

    def _run_episode(
        self,
        network: FeedForwardNetwork,
        environment: Env,
    ) -> Tuple[float, int]:
        observation = environment.reset()

        fitness = 0.0
        done = False
        step = 0
        while not done:
            if step > self._max_steps:
                break

            output = network.activate(observation)
            action = handle_action(output, environment)
            observation, reward, done, _ = environment.step(action)
            fitness += reward
            step += 1

        return fitness, step
