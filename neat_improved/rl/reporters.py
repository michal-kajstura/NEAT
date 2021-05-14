import csv
from contextlib import contextmanager
from pathlib import Path
from time import time
from typing import Iterator

_NAME = 'actor_critic'
_FIELDS = 'iteration', 'num_frames', 'time_in_s', 'fitness', 'loss'


class BaseRLReporter:
    def on_update_end(self, *args, **kwargs):
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

        with self._get_writer(_NAME, _FIELDS, 'w') as file:
            file.writeheader()

    @contextmanager
    def _get_writer(self, filename, fieldnames, mode) -> Iterator[csv.DictWriter]:
        file = (self.save_dir_path / (filename + '.csv')).open(mode)
        try:
            yield csv.DictWriter(file, fieldnames)
        finally:
            file.close()

    def on_update_end(
        self,
        num_frames,
        fitness,
        **kwargs,
    ):
        if self.start_time is None:
            self.start_time = time()

        with self._get_writer(_NAME, _FIELDS, 'a') as writer:
            writer.writerow(
                {
                    'num_frames': num_frames,
                    'fitness': fitness,
                    'time_in_s': time() - self.start_time,
                }
            )


class StdRLReporter(BaseRLReporter):
    def __init__(
        self,
        log_once_every: int = 1000,
    ):
        self.start_time = None
        self.log_once_every = log_once_every

    def on_update_end(
        self,
        num_frames,
        fitness,
        **kwargs,
    ):
        if self.start_time is None:
            self.start_time = time()

        if (num_frames % self.log_once_every) == 0:
            print(f'fitness: {fitness}')
