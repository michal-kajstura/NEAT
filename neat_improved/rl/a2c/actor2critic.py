import numpy as np
from torch import nn

from neat_improved.rl.a2c.utils import init


class ActorCritic(nn.Module):
    def __init__(self, num_inputs: int, hidden_size: int = 64):
        super(ActorCritic, self).__init__()

        self._hidden_size = hidden_size

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(layer):
        if isinstance(layer, nn.Linear):
            init(
                layer,
                nn.init.orthogonal_,
                lambda x: nn.init.constant_(x, 0), np.sqrt(2),
            )

    def forward(self, inputs):
        hidden_actor = self.actor(inputs)
        critic = self.critic(inputs)

        return critic, hidden_actor

    @property
    def output_size(self):
        return self._hidden_size
