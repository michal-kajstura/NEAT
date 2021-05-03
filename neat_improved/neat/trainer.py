import multiprocessing
from functools import wraps
from multiprocessing import Pool
from time import time
from typing import Optional, Sequence, Tuple, Callable

from neat import Config, DefaultGenome, Population
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

    def _train(self, num_frames: Optional[int], stop_time: Optional[int]) -> DefaultGenome:
        if self._num_workers is None:
            func = lambda g, c: self._evaluate_population_fitness(g, c, num_frames)
        else:
            parallel = ParallelEvaluator(
                num_workers=self._num_workers,
                evaluator=self._evaluator,
                max_num_frames=num_frames or float('inf'),
            )
            func = parallel.evaluate

        if stop_time is not None:
            # it may not be 100% reliable but it's the best we can achieve without writing a custom
            # parallel executor
            func = _timeout_func(func, time(), stop_time)

        try:
            return self._population.run(
                fitness_function=func,
                n=float('inf'),
            )
        except TimeoutError:
            return self._population.best_genome

    def _evaluate_population_fitness(
        self,
        genomes: Sequence[Tuple[int, DefaultGenome]],
        config: Config,
        max_num_frames: int,
    ):
        if self._evaluator.num_frames >= max_num_frames:
            raise TimeoutError()

        for _, genome in genomes:
            genome.fitness, num_frames = self._evaluator.evaluate(genome, config)
            self._evaluator.num_frames += num_frames


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


class ParallelEvaluator:
    def __init__(self, num_workers, evaluator: GymEvaluator, max_num_frames):
        self.num_workers = num_workers
        self.evaluator = evaluator
        self.pool = Pool(num_workers)
        self.max_num_frames = max_num_frames

    def __del__(self):
        self.pool.close()
        self.pool.join()

    def evaluate(self, genomes, config):
        if self.evaluator.num_frames >= self.max_num_frames:
            raise TimeoutError()

        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.evaluator.evaluate, (genome, config)))

        frames = 0
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness, num_frames = job.get()
            frames += num_frames

        self.evaluator.num_frames += frames

