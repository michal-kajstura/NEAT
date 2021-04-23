from collections import defaultdict
from pathlib import Path

import gym
import pandas as pd

from experiments.utils import _prepare_logging_dir
from neat_improved.rl.actor_critic import Actor, Critic, ActorCritic

EXPERIMENT_ENVS = [
    'CartPole-v0',
    # 'Pendulum-v0',
    # 'MountainCarContinuous-v0',
    'LunarLander-v2',
    # 'BipedalWalker-v3',
]

MAX_EPISODES = 1000
STOP_TIME = None
LR = 0.0001
GAMMA = 0.99
RENDER_DURING_TRAINING = False
RENDER_AFTER_TRAINING = False

LOGGING_DIR = Path('./logs_actor_critic_nn')
MODEL_SAVE_DIR = Path('../models')
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

for env_name in EXPERIMENT_ENVS:
    result_dict = defaultdict(list)

    env = gym.make(env_name).unwrapped
    logging_dir = _prepare_logging_dir(env, LOGGING_DIR)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    actor = Actor(state_size, action_size)
    critic = Critic(state_size, action_size)

    trainer = ActorCritic(
        env=env,
        actor=actor,
        critic=critic,
        n_iters=MAX_EPISODES,
        stop_time=STOP_TIME,
        render=RENDER_DURING_TRAINING,
        lr=LR,
        gamma=GAMMA,
    )
    SAVE_DIR = MODEL_SAVE_DIR / env_name
    episode_fitness_scores, episode_times = trainer.train(save_dir=SAVE_DIR)

    result_dict['episode_nr'].extend(list(range(len(episode_times))))
    result_dict['fitness'].extend(episode_fitness_scores)
    result_dict['duration'].extend(episode_times)

    result_df = pd.DataFrame.from_dict(result_dict)
    result_df['env_name'] = env_name
    result_df.to_csv(logging_dir / 'results.csv', index=False)
