import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

_NAME = 'actor_critic'
_FIELDS = 'num_frames', 'fitness', 'loss'


class CustomCallback(BaseCallback):
    def __init__(self, save_dir_path: Path, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.generation = 0
        self.save_dir_path = save_dir_path
        self.save_dir_path.mkdir(exist_ok=True, parents=True)
        self.start_time = None

        with self._get_writer(_NAME, _FIELDS, 'w') as file:
            file.writeheader()
        self._fitness_scores = [0] * 10
        self._fitness = 0.

    @contextmanager
    def _get_writer(self, filename, fieldnames, mode) -> Iterator[csv.DictWriter]:
        file = (self.save_dir_path / (filename + '.csv')).open(mode)
        try:
            yield csv.DictWriter(file, fieldnames)
        finally:
            file.close()

    def _on_step(self) -> bool:
        pass

    def _on_rollout_end(self) -> None:
        buffer = self.locals['rollout_buffer']
        rewards = buffer.rewards
        dones = buffer.dones

        episode_end_fitness_scores = []
        for rs, ds in zip(rewards, dones):
            for i, (r, d) in enumerate(zip(rs, ds)):
                if not d:
                    self._fitness_scores[i] += r
                else:
                    episode_end_fitness_scores.append(self._fitness_scores[i])
                    self._fitness_scores[i] = 0.0

        if episode_end_fitness_scores:
            self._fitness = np.array(episode_end_fitness_scores).mean()

        num_frames = self.locals['self'].num_timesteps
        with self._get_writer(_NAME, _FIELDS, 'a') as writer:
            writer.writerow(
                {
                    'num_frames': num_frames,
                    'fitness': self._fitness,
                }
            )
