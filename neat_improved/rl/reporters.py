import csv
from contextlib import contextmanager
from pathlib import Path
from time import time
from typing import Iterator

_NAME = 'actor_critic'
_FIELDS = 'iteration', 'fitness', 'time_in_s'


class BaseRLReporter:
    def on_episode_start(self, *args, **kwargs):
        pass

    def on_episode_end(self, *args, **kwargs):
        pass


class FileRLReporter(BaseRLReporter):
    def __init__(
        self,
        save_dir_path: Path,
    ):
        self.generation = 0
        self.save_dir_path = save_dir_path
        self.save_dir_path.mkdir(exist_ok=True, parents=True)
        self.start_time = None

    @contextmanager
    def _get_writer(self, filename, fieldnames, mode) -> Iterator[csv.DictWriter]:
        file = (self.save_dir_path / (filename + '.csv')).open(mode)
        try:
            yield csv.DictWriter(file, fieldnames)
        finally:
            file.close()

    def on_episode_end(self, iteration: int, fitness: float):
        if self.start_time is None:
            self.start_time = time()

        with self._get_writer(_NAME, _FIELDS, 'a') as writer:
            writer.writerow(
                {
                    'iteration': iteration,
                    'fitness': fitness,
                    'time_in_s': time() - self.start_time,
                }
            )


class StdRLReporter(BaseRLReporter):
    def __init__(self):
        self.start_time = None

    def on_episode_end(self, iteration: int, fitness: float):
        if self.start_time is None:
            self.start_time = time()

        print(f'iteration: {iteration}, fitness: {fitness}, time: {time() - self.start_time:.2f}s')
