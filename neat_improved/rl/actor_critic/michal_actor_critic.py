from collections import defaultdict
from time import time
from itertools import count
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Env
from gym.spaces import Discrete
from torch import Tensor
from torch.distributions import Categorical, Normal
from torch.optim import Adam

from neat_improved.rl.reporters import BaseRLReporter
from neat_improved.trainer import BaseTrainer


def _initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class _ContinuousOutput(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mean = nn.Linear(input_dim, output_dim)
        self.std = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        sigma = self.std(x)
        sigma = F.softplus(sigma) + 1e-5

        mean = self.mean(x)
        return mean, sigma


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_space, hidden_size=128, fit_domain_strategy='clip'):
        super().__init__()
        if fit_domain_strategy not in {'clip', 'tanh'}:
            raise ValueError(fit_domain_strategy)

        self.affine = nn.Linear(state_size, hidden_size)
        self.fit_domain_strategy = fit_domain_strategy

        self.value_layer = nn.Linear(hidden_size, 1)
        self.action_space = action_space
        if isinstance(action_space, Discrete):
            self.sample = self._sample_discrete
            self.action_layer = nn.Linear(hidden_size, action_space.n)
        else:
            self.sample = self._sample_continuous
            self.action_layer = _ContinuousOutput(hidden_size, action_space.shape[0])
            # _initialize_uniformly(self.action_layer.mean)
            # _initialize_uniformly(self.action_layer.std)
            # _initialize_uniformly(self.value_layer)


        self.logprobs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):
        state = F.relu(self.affine(state))
        state_value = self.value_layer(state)
        action_hidden = self.action_layer(state)

        action, log_prob = self.sample(action_hidden)

        self.logprobs.append(log_prob)
        self.state_values.append(state_value)

        return action

    def _sample_discrete(self, output: Tensor):
        distribution = Categorical(F.softmax(output, dim=-1))
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)

    def _sample_continuous(self, output: Tuple[Tensor, Tensor]):
        mu, sigma = output

        distribution = Normal(mu, sigma)
        action = distribution.sample()

        if self.fit_domain_strategy == 'clip':
            action = torch.clip(
                action,
                self.action_space.low[0],
                self.action_space.high[0],
            )
        else:
            action = torch.tanh(action) * (self.action_space.high[0] - self.action_space.low[0]) / 2

        return action.cpu().numpy(), distribution.log_prob(action).mean()

    def calculate_loss(self, gamma=0.99):
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)

        # normalizing the rewards:
        value, *_ = self.state_values
        rewards = torch.tensor(rewards, device=value.device, dtype=value[0].dtype)
        rewards = (rewards - rewards.mean()) / (rewards.std())

        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward - value.item()  # faktycznie gradient ma nie iść przez tę ścieżkę
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += action_loss + value_loss

        return loss

    def clear_memory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]


class ActorCriticTrainer(BaseTrainer):
    def __init__(
        self,
        env: Env,
        actor_critic: nn.Module,
        lr: float = 0.001,
        gamma: float = 0.99,
        render: bool = False,
        save_dir: Optional[Path] = None,
        use_gpu: bool = True,
        reporters: Optional[Sequence[BaseRLReporter]] = None,
    ):
        self.lr = lr
        self.gamma = gamma
        self.render = render
        self.save_dir = save_dir
        self.reporters = reporters or ()

        self.device = torch.device('cuda:0' if use_gpu else 'cpu')
        self.actor_critic = actor_critic.to(self.device)
        self.history = defaultdict(list)
        self.env = env
        self.optimizer = Adam(
            params=actor_critic.parameters(),
            lr=lr,
        )

    def _train(self, iterations: Optional[int], stop_time: Optional[int]):
        start_time = time()
        iter_ = count() if iterations is None else range(iterations)

        for iteration in iter_:
            if stop_time and (time() - start_time) >= stop_time:
                break

            fitness = self._train_episode()
            self._call_reporters(
                'on_episode_end',
                iteration=iteration,
                fitness=fitness,
            )

        self.env.close()

    def _train_episode(self):
        state = self.env.reset()
        fitness = 0.0

        for _ in count():
            if self.render:
                self.env.render()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            action = self.actor_critic(state)
            state, reward, done, _ = self.env.step(action)

            self.actor_critic.rewards.append(reward)
            fitness += reward

            if done:
                break

        self.optimizer.zero_grad()
        loss = self.actor_critic.calculate_loss(self.gamma)
        loss.backward()
        self.optimizer.step()
        self.actor_critic.clear_memory()

        return fitness

    def _compute_returns(self, rewards):
        total_reward = 0
        returns = []
        for reward in reversed(rewards):
            total_reward = reward + self.gamma * total_reward
            returns.insert(0, total_reward)
        return returns

    def _call_reporters(self, stage: str, *args, **kwargs):
        for reporter in self.reporters:
            getattr(reporter, stage)(*args, **kwargs)
