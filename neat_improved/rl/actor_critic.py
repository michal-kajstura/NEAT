from collections import defaultdict
from itertools import count
from pathlib import Path
from time import time
from typing import List, Optional, Tuple

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import Env
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


class ActorCritic:
    def __init__(
            self,
            env: Env,
            actor: nn.Module,
            critic: nn.Module,
            stop_time: Optional[time] = None,
            n_iters: Optional[int] = None,
            lr: float = 0.001,
            gamma: float = 0.99,
            render: bool = False,
    ):
        self.lr = lr
        self.gamma = gamma
        self.render = render
        self.n_iters = n_iters
        self.stop_time = stop_time

        if not self.n_iters and not self.stop_time:
            self.n_iters = 1000

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = defaultdict(list)
        self.env = env
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.optimizers = {
            'actor': optim.Adam(self.actor.parameters(), lr=self.lr),
            'critic': optim.Adam(self.critic.parameters(), lr=self.lr),
        }

    def train(self, save_dir: Optional[Path] = None) -> Tuple[List, List]:
        train_duration = 0.0
        episode_fitness_scores = []
        episode_times = []
        iter_ = count() if not self.n_iters else range(self.n_iters)

        for num in iter_:
            if self.stop_time and train_duration >= self.stop_time:
                break

            start_time = time()
            fitness = self._train_episode(num)
            stop_time = time() - start_time

            train_duration += stop_time
            episode_fitness_scores.append(fitness)
            episode_times.append(stop_time)

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.actor, save_dir.joinpath('actor.pkl'))
            torch.save(self.critic, save_dir.joinpath('critic.pkl'))

        self.env.close()
        return episode_fitness_scores, episode_times

    def _train_episode(self, num: int):
        state = self.env.reset()
        episode = defaultdict(list)
        fitness = 0.0

        for i in count():
            if self.render:
                self.env.render()
            state = torch.FloatTensor(state).to(self.device)
            dist = self.actor(state)
            value = self.critic(state)

            action = dist.sample()
            state, reward, done, _ = self.env.step(action.cpu().numpy())
            fitness += reward
            log_prob = dist.log_prob(action).unsqueeze(0)

            episode['log_probs'].append(log_prob)
            episode['values'].append(value)
            episode['rewards'].append(reward)

            if done:
                print('Iteration: {}, Score: {}'.format(num, i))
                break

        returns = self._compute_returns(episode['rewards'])

        log_probs = torch.cat(episode['log_probs'])
        values = torch.cat(episode['values'])
        returns = torch.FloatTensor(returns).to(self.device).detach()

        advantage = returns - values

        losses = {
            'actor': -(log_probs * advantage.detach()).mean(),
            'critic': advantage.pow(2).mean(),
        }

        for name, optimizer in self.optimizers.items():
            optimizer.zero_grad()
            losses[name].backward()
            optimizer.step()

        return fitness

    def _compute_returns(self, rewards):
        total_reward = 0
        returns = []
        for reward in reversed(rewards):
            total_reward = reward + self.gamma * total_reward
            returns.insert(0, total_reward)
        return returns


if __name__ == '__main__':
    env = gym.make("CartPole-v0").unwrapped

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    lr = 0.0001

    actor = Actor(state_size, action_size)
    critic = Critic(state_size, action_size)

    trainer = ActorCritic(env, actor, critic, n_iters=1000)
    trainer.train(save_dir=Path('../models'))
