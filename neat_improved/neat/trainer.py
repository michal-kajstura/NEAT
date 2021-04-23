import multiprocessing
from functools import wraps
from time import time
from typing import Optional, Sequence, Tuple, Callable

from neat import Config, DefaultGenome, ParallelEvaluator, Population
from neat.reporting import BaseReporter

from neat_improved.neat.evaluator import GymEvaluator
from neat_improved.trainer import BaseTrainer


class NEATRunner(BaseTrainer):
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

    def _train(self, generations: Optional[int], stop_time: Optional[int]) -> DefaultGenome:
        if self._num_workers is None:
            func = self._evaluate_population_fitness
            if stop_time is not None:
                func = _timeout_func(func, time(), stop_time)
        else:
            parallel = ParallelEvaluator(
                num_workers=self._num_workers,
                eval_function=self._evaluator.evaluate,
                timeout=stop_time,
            )
            func = parallel.evaluate

        try:
            return self._population.run(
                fitness_function=func,
                n=generations or float('inf'),
            )
        except TimeoutError:
            return self._population.best_genome

    def _evaluate_population_fitness(
        self,
        genomes: Sequence[Tuple[int, DefaultGenome]],
        config: Config,
    ):
        for _, genome in genomes:
            genome.fitness = self._evaluator.evaluate(genome, config)


def _timeout_func(
    func: Callable,
    start_time,
    stop_time,
):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if (time() - start_time) >= stop_time:
            raise TimeoutError()

        return func(*args, **kwargs)

    return wrapper
