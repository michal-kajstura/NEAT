from datetime import datetime
from pathlib import Path
from typing import Optional

from gym import Env
from neat import Config, StatisticsReporter, StdOutReporter
from neat.nn import FeedForwardNetwork

from neat_improved.action_handler import handle_action
from neat_improved.evaluator import MultipleRunGymEvaluator
from neat_improved.reporters import FileReporter
from neat_improved.runner import NEATRunner


def render_result(
        environment: Env,
        network: FeedForwardNetwork,
        steps: int = 500,
):
    frames = []
    observation = environment.reset()
    for _ in range(steps):
        outputs = network.activate(observation)
        action = handle_action(outputs, environment)
        observation, _, done, _ = environment.step(action)
        frame = environment.render(mode='rgb_array')
        frames.append(frame)

        if done:
            break

    environment.close()

    return frames


def run(
        environment: Env,
        config: Config,
        num_generations: int,
        num_workers: Optional[int] = None,
        num_repeats: int = 1,
        logging_root: Optional[Path] = None,
):
    evaluator = MultipleRunGymEvaluator(
        environment=environment,
        max_steps=1000,
        runs_per_network=3,
        render=False,
    )

    best_genomes = [
        _run(
            evaluator=evaluator,
            config=config,
            num_generations=num_generations,
            logging_root=logging_root,
            num_workers=num_workers,
        )
        for _ in range(num_repeats)
    ]

    return best_genomes[-1]


def _run(
        evaluator: MultipleRunGymEvaluator,
        config: Config,
        num_generations: int,
        logging_root: Optional[Path] = None,
        num_workers: Optional[int] = None,
):
    logging_dir = _prepare_logging_dir(
        evaluator._environment,
        logging_root or Path('./logs'),
    )
    runner = NEATRunner(
        config=config,
        evaluator=evaluator,
        reporters=[
            StatisticsReporter(),
            StdOutReporter(show_species_detail=False),
            FileReporter(save_dir_path=logging_dir),
        ],
        num_workers=num_workers,
    )

    best_genome = runner.run(num_generations)
    return best_genome


def _prepare_logging_dir(
        environment: Env,
        root: Path,
) -> Path:
    now = datetime.now()
    time = now.strftime('%Y-%m-%d %H:%M:%S')

    env_name = environment.spec._env_name
    logging_dir = root / env_name / time
    logging_dir.mkdir(exist_ok=True, parents=True)

    return logging_dir
