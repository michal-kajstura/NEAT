import abc
from typing import Optional


class BaseTrainer(abc.ABC):
    def train(self, iterations: Optional[int], stop_time: Optional[int]):
        if (iterations is None) == (stop_time is None):
            raise ValueError('Both iterations and stop_time are set to None')

        self._train(iterations, stop_time)

    @abc.abstractmethod
    def _train(self, iterations: Optional[int], stop_time: Optional[int]):
        pass
