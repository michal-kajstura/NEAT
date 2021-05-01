from typing import Sequence, Type

import numpy as np
from torch import nn

from neat_improved.rl.actor_critic.distributions import get_action_distribution
from neat_improved.rl.actor_critic.utils import init


def _create_mlp(
    channels: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    layers = []
    for in_ch, out_ch in zip(channels, channels[1:]):
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

        self._hidden_size = hidden_size

        self.actor = _create_mlp(
            channels=(num_inputs, *([hidden_size] * num_hidden_layers)),
            activation_class=nn.Tanh,
        )

        self.critic = _create_mlp(
            channels=(num_inputs, *([hidden_size] * num_hidden_layers), 1),
            activation_class=nn.Tanh,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(layer):
        if isinstance(layer, nn.Linear):
            init(
                layer,
                weight_init=nn.init.orthogonal_,
                bias_init=lambda x: nn.init.constant_(x, 0),
                gain=np.sqrt(2),
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
    def __init__(self, obs_shape, action_space, common_stem=False):
        super(PolicyA2C, self).__init__()

        cls = CommonStemActorCritic if common_stem else ActorCritic
        self.actor_critic = cls(num_inputs=obs_shape[0])
        self.dist = get_action_distribution(action_space, self.actor_critic.output_size)

    def forward(self, inputs):
        critic_values, actor_features = self.actor_critic(inputs)
        dist = self.dist(actor_features)

        # TODO: maybe allow deterministic
        action = dist.sample()
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()

        return action, critic_values, action_log_probs, dist_entropy

    def get_critic_values(self, inputs):
        critic_values, _ = self.actor_critic(inputs)
        return critic_values
