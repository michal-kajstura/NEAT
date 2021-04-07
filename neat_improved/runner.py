import multiprocessing
from typing import Sequence, Optional

from neat import Config, Population, ParallelEvaluator, DefaultGenome
from neat.reporting import BaseReporter

from neat_improved.evaluator import GymEvaluator


class NEATRunner:
    def __init__(
        self,
        config: Config,
        evaluator: GymEvaluator,
        reporters: Optional[Sequence[BaseReporter]] = None,
        num_workers: int = multiprocessing.cpu_count(),
    ):
        self._population = Population(config)

        reporters = reporters or []
        for reporter in reporters:
            self._population.add_reporter(reporter)

        self._parallel = ParallelEvaluator(
            num_workers=num_workers,
            eval_function=evaluator.evaluate,
        )

    def run(
        self,
        generations: int,
    ) -> DefaultGenome:
        return self._population.run(
            fitness_function=self._parallel.evaluate,
            n=generations,
        )
