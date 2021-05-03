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
        iteration,
        num_frames,
        fitness,
        policy_loss,
        **kwargs,
    ):
        if self.start_time is None:
            self.start_time = time()

        with self._get_writer(_NAME, _FIELDS, 'a') as writer:
            writer.writerow(
                {
                    'iteration': iteration,
                    'num_frames': num_frames,
                    'fitness': fitness,
                    'loss': policy_loss,
                    'time_in_s': time() - self.start_time,
                }
            )


class StdRLReporter(BaseRLReporter):
    def __init__(
        self,
        log_once_every: int = 1,
    ):
        self.start_time = None
        self.log_once_every = log_once_every

    def on_update_end(
        self,
        iteration,
        num_frames,
        fitness,
        policy_loss,
        **kwargs,
    ):
        if self.start_time is None:
            self.start_time = time()

        if (iteration % self.log_once_every) == 0:
            print(
                f'iteration: {iteration}, fitness: {fitness}, time: {time() - self.start_time:.2f}s'
                f' loss: {policy_loss}'
            )
