import abc
from typing import Optional


class BaseTrainer(abc.ABC):
    def train(self, num_frames: Optional[int], stop_time: Optional[int]):
        if (num_frames is None) == (stop_time is None):
            raise ValueError('Both iterations and stop_time are set to None')

        self._train(num_frames, stop_time)

    @abc.abstractmethod
    def _train(self, num_frames: Optional[int], stop_time: Optional[int]):
        pass
