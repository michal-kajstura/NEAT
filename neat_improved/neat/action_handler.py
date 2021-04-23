from typing import Any

import numpy as np
from gym import Env
from gym.spaces import Discrete


def handle_action(
    output: Any,
    environment: Env,
) -> Any:
    if isinstance(environment.action_space, Discrete):
        return np.argmax(output)
    return output
