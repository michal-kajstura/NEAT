import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from gym import Env
from neat import Config, StatisticsReporter, StdOutReporter
from neat.nn import FeedForwardNetwork

from neat_improved.neat.action_handler import handle_action
from neat_improved.neat.evaluator import MultipleRunGymEvaluator
from neat_improved.neat.reporters import FileReporter
from neat_improved.neat.trainer import NEATRunner
from neat_improved.rl.actor_critic.michal_actor_critic import ActorCritic, ActorCriticTrainer
from neat_improved.rl.reporters import StdRLReporter, FileRLReporter


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


def run_neat(
    environment: Env,
    config: Config,
    num_generations: Optional[int],
    stop_time: Optional[int],
    num_workers: Optional[int] = None,
    runs_per_network: int = 1,
    max_steps: int = 1000,
    logging_root: Optional[Path] = None,
):
    evaluator = MultipleRunGymEvaluator(
        environment=environment,
        max_steps=max_steps,
        runs_per_network=runs_per_network,
    )

    logging_dir = _prepare_logging_dir(
        environment,
        logging_root or Path('./logs'),
    )

    with (logging_dir / 'hyperparameters.json').open('w') as file:
        json.dump(
            {
                'num_generations': num_generations,
                'stop_time': stop_time,
                'num_workers': num_workers,
                'runs_per_network': runs_per_network,
                'max_steps': max_steps,
            },
            file,
            indent=4,
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

    best_genome = runner.train(num_generations, stop_time)
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


def run_actor_critic(
    environment: Env,
    num_iterations: Optional[int],
    lr: float,
    gamma: float,
    stop_time: Optional = None,
    use_gpu: bool = True,
    logging_root: Optional[Path] = None,
):
    logging_dir = _prepare_logging_dir(environment, logging_root or Path('./ac_results'))

    with (logging_dir / 'hyperparameters.json').open('w') as file:
        json.dump(
            {
                'num_iterations': num_iterations,
                'stop_time': stop_time,
                'lr': lr,
                'gamma': gamma,
                'use_gpu': use_gpu,
            },
            file,
            indent=4,
        )

    state_size = environment.observation_space.shape[0]

    trainer = ActorCriticTrainer(
        env=environment,
        actor_critic=ActorCritic(
            state_size=state_size,
            action_space=environment.action_space,
            fit_domain_strategy='tanh',
        ),
        render=False,
        lr=lr,
        gamma=gamma,
        save_dir=logging_dir,
        reporters=(
            StdRLReporter(
                log_once_every=10,
            ),
            FileRLReporter(
                save_dir_path=logging_dir,
            ),
        ),
        use_gpu=use_gpu,
    )
    trainer.train(num_iterations, stop_time)
