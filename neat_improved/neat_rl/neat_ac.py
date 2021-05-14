from collections import defaultdict
from typing import Optional, Callable

import neat
import numpy as np
import torch
import torch.nn.functional as F
from gym import Env
from gym.spaces import Box
from neat.nn import FeedForwardNetwork
from torch import optim

from neat_improved.rl.actor_critic.a2c import PolicyA2C
from neat_improved.trainer import BaseTrainer


class NEATACNetwork:
    def __init__(self, neat_network, dim, action_space):
        self._policy = PolicyA2C(
            dim,
            action_space,
            actor_critic_kwargs={'num_hidden_layers': 0, 'hidden_size': dim[0]},
        )
        self._neat_network = neat_network
        self.optimizer = optim.RMSprop(self._policy.parameters(), lr=7e-4)

    def forward(self, state):
        state_features = self._neat_network.activate(state)
        state_features = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
        # action, critic_values, action_log_probs, dist_entropy = self._policy(state_features)
        return self._policy(state_features)


class NEATACTrainer(BaseTrainer):
    def __init__(
        self,
        env: Env,
        config: neat.Config,
        population_size: int = 10,
        gamma: float = 0.99,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        normalize_advantage: bool = False,
        use_gpu: bool = True,
        critic_loss_func: Callable = F.mse_loss,
    ):
        dim = 10
        config.genome_config.num_outputs = dim
        config.genome_config.output_keys = list(range(dim))
        config.pop_size = 5
        self.config = config
        self.obs_shape = env.observation_space.shape

        self.population = neat.Population(config)
        self.device = 'cuda' if use_gpu else 'cpu'

        self.env = env
        self.action_space = self.env.action_space

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage

        self.critic_loss_func = critic_loss_func

    def _train(self, num_frames: Optional[int], stop_time: Optional[int]):
        self.population.run(self._evaluate_fitness, 500)

    def _evaluate_fitness(self, genomes, config):
        genomes = [genome for _, genome in genomes]
        networks = self._init_networks(genomes, config)

        for genome, network in zip(genomes, networks):
            fitnesses = [self._episode(network) for _ in range(200)]
            fitness = np.mean(fitnesses)
            print(fitness)
            genome.fitness = fitness

    def _episode(self, network: NEATACNetwork):
        state = self.env.reset()
        fitness = 0

        buffer = defaultdict(list)
        for step in range(10000):
            action, critic_values, action_log_probs, dist_entropy = network.forward(state)
            action = action.cpu().numpy()[0]

            # Clip the actions to avoid out of bound error
            clipped_action = action
            if isinstance(self.action_space, Box):
                clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
            else:
                clipped_action = clipped_action.flatten()[0]

            state, reward, done, _ = self.env.step(clipped_action)
            state = state.flatten()

            fitness += reward

            buffer['log_probs'].append(action_log_probs)
            buffer['values'].append(critic_values)
            buffer['rewards'].append(reward)

            if done:
                break

        log_probs = torch.stack(buffer['log_probs'])
        values = torch.stack(buffer['values']).squeeze()
        returns = self._compute_returns(buffer['rewards'])
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        advantage = returns - values

        policy_loss = torch.mean(-log_probs * advantage.detach())
        value_loss = F.smooth_l1_loss(values, returns)

        network.optimizer.zero_grad()
        loss = policy_loss + 0.5 * value_loss

        loss.backward()
        network.optimizer.step()

        print(loss)
        return fitness

    def _compute_returns(self, rewards):
        r = 0.
        returns = []
        for step in reversed(range(len(rewards))):
            r = rewards[step] + self.gamma * r
            returns.insert(0, r)

        return torch.tensor(returns)

    def _init_networks(self, genomes, config):
        networks = []
        for genome in genomes:
            network = FeedForwardNetwork.create(genome, config)
            neat_ac_network = NEATACNetwork(
                network,
                [self.config.genome_config.num_outputs],
                self.action_space,
            )
            networks.append(neat_ac_network)
        return networks
