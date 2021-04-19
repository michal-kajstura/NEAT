from collections import defaultdict
from itertools import count
from pathlib import Path

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


class Trainer:
    def __init__(self, env, actor, critic, n_iters, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = defaultdict(list)
        self.env = env
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.n_iters = n_iters
        self.gamma = gamma
        self.optimizers = {
            'actor': optim.Adam(self.actor.parameters()),
            'critic': optim.Adam(self.critic.parameters()),
        }

    def train(self, save_dir: Path):
        for num in range(self.n_iters):
            self._train_episode(num)

        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor, save_dir.joinpath('actor.pkl'))
        torch.save(self.critic, save_dir.joinpath('critic.pkl'))
        self.env.close()

    def _train_episode(self, num):
        state = self.env.reset()
        episode = defaultdict(list)

        for i in count():
            self.env.render()
            state = torch.FloatTensor(state).to(self.device)
            dist = self.actor(state)
            value = self.critic(state)

            action = dist.sample()
            state, reward, done, _ = self.env.step(action.cpu().numpy())
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

    trainer = Trainer(env, actor, critic, n_iters=1000)
    trainer.train(save_dir=Path('../models'))
