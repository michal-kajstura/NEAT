from functools import partial
from typing import Sequence, Type

import numpy as np
import torch
from stable_baselines3.common.distributions import (
    make_proba_distribution,
    DiagGaussianDistribution,
    CategoricalDistribution,
    Distribution,
)
from torch import nn


def _create_mlp(
    channels: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    layers = []
    for i in range(0, len(channels) - 1):
        in_ch = channels[i]
        out_ch = channels[i + 1]
        layers.append(nn.Linear(in_ch, out_ch))
        layers.append(activation_class())

    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.actor = _create_mlp(
            channels=(num_inputs, *([hidden_size] * num_hidden_layers)),
            activation_class=nn.Tanh,
        )

        self.critic = _create_mlp(
            channels=(num_inputs, *([hidden_size] * num_hidden_layers)),
            activation_class=nn.Tanh,
        )

    def forward(self, inputs):
        hidden_actor = self.actor(inputs)
        critic = self.critic(inputs)
        return critic, hidden_actor

    @property
    def output_size(self):
        return self._hidden_size


class CommonStemActorCritic(ActorCritic):
    def __init__(
        self,
        num_inputs: int,
        hidden_size: int = 64,
    ):
        super().__init__(
            num_inputs=hidden_size,
            hidden_size=hidden_size,
            num_hidden_layers=1,
        )

        self.feature_extractor = _create_mlp((num_inputs, hidden_size), nn.Tanh)
        self.apply(self._init_weights)

    def forward(self, inputs):
        features = self.feature_extractor(inputs)
        return super().forward(features)


class PolicyA2C(nn.Module):
    def __init__(self, obs_shape, action_space, common_stem=False, actor_critic_kwargs=None):
        super(PolicyA2C, self).__init__()

        cls = CommonStemActorCritic if common_stem else ActorCritic
        self.actor_critic = cls(num_inputs=obs_shape[0], **(actor_critic_kwargs or {}))
        self.action_dist = make_proba_distribution(action_space)

        hidden_size = self.actor_critic.hidden_size
        self.value_net = nn.Linear(hidden_size, 1)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=hidden_size,
                log_std_init=0.0,
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=hidden_size)
        else:
            raise ValueError()

        # originally from openai/baselines (default gains/init_scales).
        module_gains = {
            self.actor_critic: np.sqrt(2),
            self.action_net: 0.01,
            self.value_net: 1,
        }
        for module, gain in module_gains.items():
            module.apply(partial(self.init_weights, gain=gain))

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, inputs):
        critic_features, actor_features = self.actor_critic(inputs)

        values = self.value_net(critic_features)
        distribution = self._get_action_dist_from_latent(actor_features)
        action = distribution.get_actions()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return action, values, log_prob, entropy

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError()
