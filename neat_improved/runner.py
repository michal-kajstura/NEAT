import multiprocessing
from typing import Sequence, Optional, List, Tuple

from neat import Config, Population, ParallelEvaluator, DefaultGenome
from neat.reporting import BaseReporter

from neat_improved.evaluator import GymEvaluator


class NEATRunner:
    def __init__(
        self,
        config: Config,
        evaluator: GymEvaluator,
        reporters: Optional[Sequence[BaseReporter]] = None,
        num_workers: Optional[int] = multiprocessing.cpu_count(),
    ):
        self._evaluator = evaluator

        self._population = Population(config)

        reporters = reporters or []
        for reporter in reporters:
            self._population.add_reporter(reporter)

        self._num_workers = num_workers

    def run(
        self,
        generations: int,
    ) -> DefaultGenome:
        if self._num_workers is None:
            func = self._evaluate_population_fitness
        else:
            parallel = ParallelEvaluator(
                num_workers=self._num_workers,
                eval_function=self._evaluator.evaluate,
            )
            func = parallel.evaluate

        return self._population.run(
            fitness_function=func,
            n=generations,
        )

    def _evaluate_population_fitness(
        self,
        genomes: Sequence[Tuple[int, DefaultGenome]],
        config: Config,
    ):
        for _, genome in genomes:
            genome.fitness = self._evaluator.evaluate(genome, config)
