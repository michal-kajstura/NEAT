from pathlib import Path

import gym
import torch
from gym.spaces import Box
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

from experiments.utils import _prepare_logging_dir
from neat_improved.rl.actor_critic.a2c import PolicyA2C
from neat_improved.rl.actor_critic.trainer import A2CTrainer

EXPERIMENT_ENVS = [
    # 'CartPole-v0',
    # 'Pendulum-v0',
    # 'MountainCarContinuous-v0',
    'LunarLander-v2',
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
    envs = make_vec_env(
        env_id=env_name,
        seed=SEED,
        n_envs=5,
        monitor_dir=str(logging_dir),
        vec_env_cls=ENV_WRAPPER_CLS,
    )

    policy = PolicyA2C(envs.observation_space.shape, envs.action_space)
    trainer = A2CTrainer(
        policy=policy,
        vec_envs=envs,
        n_steps=5,
        use_gpu=USE_CUDA,
        log_interval=LOG_INTERVAL,
        lr=0.0007,
        normalize_advantage=False,
    )

    trainer.train(stop_time=60 * 2, iterations=None)

    device = torch.device('cuda')
    policy = trainer.policy
    state = envs.reset()
    while True:
        state = torch.tensor(state, dtype=torch.float32, device=device)
        action, critic_values, action_log_probs, dist_entropy = policy(state)
        action = action.cpu().numpy()

        # Clip the actions to avoid out of bound error
        clipped_action = action
        if isinstance(env.action_space, Box):
            clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
        else:
            clipped_action = clipped_action.flatten()

        # take action in env and look the results
        state, reward, done, infos = envs.step(clipped_action)

        obs, rewards, dones, info = envs.step(clipped_action)
        envs.render()

