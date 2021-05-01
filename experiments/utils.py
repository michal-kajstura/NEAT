import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import gym
from gym import Env
from neat import Config, StatisticsReporter, StdOutReporter
from neat.nn import FeedForwardNetwork
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from neat_improved.neat.action_handler import handle_action
from neat_improved.neat.evaluator import MultipleRunGymEvaluator
from neat_improved.neat.reporters import FileReporter
from neat_improved.neat.trainer import NEATRunner
from neat_improved.rl.actor_critic.a2c import PolicyA2C
from neat_improved.rl.actor_critic.trainer import A2CTrainer


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
    environment_name: str,
    config: Config,
    num_generations: Optional[int],
    stop_time: Optional[int],
    num_workers: Optional[int] = None,
    runs_per_network: int = 1,
    max_steps: int = 1000,
    logging_root: Optional[Path] = None,
    seed: int = 2021,
):
    environment = gym.make(environment_name)
    evaluator = MultipleRunGymEvaluator(
        environment=environment,
        max_steps=max_steps,
        runs_per_network=runs_per_network,
    )

    logging_dir = _prepare_logging_dir(
        environment_name,
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
                'seed': seed,
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

    runner.train(num_generations, stop_time)


def _prepare_logging_dir(
    env_name: str,
    root: Path,
) -> Path:
    now = datetime.now()
    time = now.strftime('%Y-%m-%d %H:%M:%S')

    logging_dir = root / env_name / time
    logging_dir.mkdir(exist_ok=True, parents=True)

    return logging_dir


def run_actor_critic(
    environment_name: str,
    num_iterations: Optional[int],
    lr: float,
    gamma: float,
    stop_time: Optional = None,
    use_gpu: bool = True,
    logging_root: Optional[Path] = None,
    normalize_advantage: bool = False,
    value_loss_coef: float = 0.5,
    seed=2021,
):
    logging_dir = _prepare_logging_dir(environment_name, logging_root or Path('./ac_results'))

    with (logging_dir / 'hyperparameters.json').open('w') as file:
        json.dump(
            {
                'num_iterations': num_iterations,
                'stop_time': stop_time,
                'lr': lr,
                'gamma': gamma,
                'normalize_advantage': normalize_advantage,
                'value_loss_coef': value_loss_coef,
                'use_gpu': use_gpu,
                'seed': seed,
            },
            file,
            indent=4,
        )

    envs = make_vec_env(
        env_id=environment_name,
        seed=seed,
        n_envs=5,
        monitor_dir=str(logging_dir),
        vec_env_cls=DummyVecEnv,
    )

    policy = PolicyA2C(
        envs.observation_space.shape,
        envs.action_space,
        common_stem=False,
    )
    trainer = A2CTrainer(
        policy=policy,
        vec_envs=envs,
        n_steps=5,
        use_gpu=use_gpu,
        log_interval=10,
        value_loss_coef=value_loss_coef,
        lr=lr,
        normalize_advantage=normalize_advantage,
    )

    trainer.train(stop_time=stop_time, iterations=num_iterations,)
