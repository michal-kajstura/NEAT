import abc
import csv
from contextlib import contextmanager
from pathlib import Path
from time import time
from typing import Iterator

from neat.reporting import BaseReporter

_SPECIES = 'species'
_POPULATION = 'population'
_FIELDS = {
    _SPECIES: (
        'iteration',
        'species_id',
        'size',
        'age',
        'stagnation',
        'fitness',
        'adjusted_fitness',
        'time_in_s',
    ),
    _POPULATION: (
        'iteration',
        'individual_id',
        'species_id',
        'fitness',
        'num_nodes',
        'num_connections',
        'time_in_s',
    ),
}


class FileReporter(BaseReporter):
    def __init__(
        self,
        save_dir_path: Path,
    ):
        self.generation = 0
        self.start_time = None
        self.save_dir_path = save_dir_path
        self.save_dir_path.mkdir(exist_ok=True, parents=True)

        for key, fieldnames in _FIELDS.items():
            with self._get_writer(key, fieldnames, 'w') as file:
                file.writeheader()

    @contextmanager
    def _get_writer(self, filename, fieldnames, mode) -> Iterator[csv.DictWriter]:
        file = (self.save_dir_path / (filename + '.csv')).open(mode)
        try:
            yield csv.DictWriter(file, fieldnames)
        finally:
            file.close()

    def end_generation(self, config, population, species_set):
        self.generation += 1

    def post_evaluate(self, config, population, species, best_genome):
        if self.start_time is None:
            self.start_time = time()

        with self._get_writer(_POPULATION, _FIELDS[_POPULATION], 'a') as writer:
            for key, individual in population.items():
                species_id = species.get_species_id(key)
                writer.writerow(
                    {
                        'iteration': self.generation,
                        'individual_id': key,
                        'species_id': species_id,
                        'fitness': individual.fitness,
                        'num_nodes': len(individual.nodes),
                        'num_connections': len(individual.connections),
                        'time_in_s': time() - self.start_time,
                    }
                )

        with self._get_writer(_SPECIES, _FIELDS[_SPECIES], 'a') as writer:
            for key, specie in species.species.items():
                writer.writerow(
                    {
                        'iteration': self.generation,
                        'species_id': key,
                        'size': len(specie.members),
                        'age': self.generation - specie.created,
                        'stagnation': self.generation - specie.last_improved,
                        'fitness': specie.fitness,
                        'adjusted_fitness': specie.adjusted_fitness,
                        'time_in_s': time() - self.start_time,
                    }
                )
