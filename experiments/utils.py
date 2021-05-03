import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import neat
import torch
from gym import Env
from neat import StdOutReporter
from neat.nn import FeedForwardNetwork
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from neat_improved.neat import NEAT_CONFIGS
from neat_improved.neat.action_handler import handle_action
from neat_improved.neat.evaluator import MultipleRunGymEvaluator
from neat_improved.neat.reporters import FileReporter
from neat_improved.neat.trainer import NEATRunner
from neat_improved.rl.actor_critic.a2c import PolicyA2C
from neat_improved.rl.actor_critic.callbacks import CustomCallback
from neat_improved.rl.actor_critic.trainer import A2CTrainer
from neat_improved.rl.reporters import FileRLReporter


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
    max_frames: Optional[int],
    stop_time: Optional[int],
    logging_dir: Path,
    num_workers: Optional[int] = None,
    runs_per_network: int = 1,
    max_steps: int = 1000,
    seed: int = 2021,
):
    logging_dir = prepare_logging_dir(environment_name, logging_dir)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(NEAT_CONFIGS[environment_name]),
    )

    evaluator = MultipleRunGymEvaluator(
        environment_name=environment_name,
        max_steps=max_steps,
        runs_per_network=runs_per_network,
    )


    with (logging_dir / 'hyperparameters.json').open('w') as file:
        json.dump(
            {
                'max_frames': max_frames,
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
            # StatisticsReporter(),
            StdOutReporter(show_species_detail=False),
            FileReporter(save_dir_path=logging_dir, evaluator=evaluator),
        ],
        num_workers=num_workers,
    )

    runner.train(max_frames, stop_time)


def prepare_logging_dir(
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
    max_frames: Optional[int],
    lr: float,
    gamma: float,
    logging_dir: Path,
    stop_time: Optional = None,
    use_gpu: bool = True,
    normalize_advantage: bool = False,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    common_stem: bool = False,
    seed=2021,
):

    logging_dir = prepare_logging_dir(environment_name, logging_dir)
    with (logging_dir / 'hyperparameters.json').open('w') as file:
        json.dump(
            {
                'max_frames': max_frames,
                'stop_time': stop_time,
                'lr': lr,
                'gamma': gamma,
                'normalize_advantage': normalize_advantage,
                'value_loss_coef': value_loss_coef,
                'common_stem': common_stem,
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
        # vec_env_cls=DummyVecEnv,
        vec_env_cls=SubprocVecEnv,
    )

    policy = PolicyA2C(
        envs.observation_space.shape,
        envs.action_space,
        common_stem=common_stem,
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
        entropy_coef=entropy_coef,
        reporters=(FileRLReporter(save_dir_path=logging_dir),),
    )

    trainer.train(
        stop_time=stop_time,
        num_frames=max_frames,
    )


def run_baseline_actor_critic(
    environment_name: str,
    max_frames: Optional[int],
    logging_dir: Path,
    seed=2021,
):
    logging_dir = prepare_logging_dir(environment_name, logging_dir)

    envs = make_vec_env(
        env_id=environment_name,
        seed=seed,
        n_envs=5,
        monitor_dir=str(logging_dir),
    )

    a2c = A2C(
        policy=MlpPolicy,
        env=envs,
        verbose=1,
        device=torch.device('cuda'),
    )
    a2c.learn(max_frames, callback=CustomCallback(logging_dir))

