from pathlib import Path

import gym
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from experiments.utils import _prepare_logging_dir
from neat_improved.rl.a2c_new.trainer import A2CTrainer

EXPERIMENT_ENVS = [
    'CartPole-v0',
    # 'Pendulum-v0',
    # 'MountainCarContinuous-v0',
    # 'LunarLander-v2',
    # 'BipedalWalker-v3',
]

USE_CUDA = True
LOGGING_DIR = Path('./logs_actor_critic_nn')
SEED = 2021
ENV_NUM = 10
ENV_WRAPPER_CLS = DummyVecEnv
FORWARD_STEPS = 5
TOTAL_STEPS = int(10e6)
LOG_INTERVAL = 100

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(1)

for env_name in EXPERIMENT_ENVS:
    env = gym.make(env_name)
    logging_dir = _prepare_logging_dir(env, LOGGING_DIR)

    # envs = make_env_vec(env_name=env_name, num_envs=ENV_NUM)
    envs = make_vec_env(env_id=env_name, seed=SEED, n_envs=ENV_NUM, monitor_dir=str(logging_dir),
                        vec_env_cls=ENV_WRAPPER_CLS)

    trainer = A2CTrainer(
        vec_envs=envs,
        n_steps=FORWARD_STEPS,
        use_gpu=USE_CUDA,
        log_interval=LOG_INTERVAL,
    )

    trainer._train(TOTAL_STEPS, stop_time=None)
