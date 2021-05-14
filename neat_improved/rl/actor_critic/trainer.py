from itertools import count
from time import time
from typing import Optional, Sequence, Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import Box
from stable_baselines3.common.vec_env import VecEnv
from torch import nn, optim, Tensor

from neat_improved.rl.actor_critic.a2c import PolicyA2C
from neat_improved.rl.reporters import BaseRLReporter
from neat_improved.trainer import BaseTrainer

_FLOAT = torch.float32


class A2CTrainer(BaseTrainer):
    def __init__(
        self,
        policy: PolicyA2C,
        vec_envs: VecEnv,
        n_steps: int = 5,
        lr: float = 7e-4,
        lr_scheduler: Optional[str] = None,
        eps: float = 1e-5,
        alpha: float = 0.99,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        log_interval: int = 10000,
        normalize_advantage: bool = False,
        use_gpu: bool = True,
        critic_loss_func: Callable = F.mse_loss,
        reporters: Optional[Sequence[BaseRLReporter]] = None,
    ):
        super().__init__()

        self.device = 'cuda' if use_gpu else 'cpu'
        self.policy = policy.to(self.device)

        self.vec_envs = vec_envs
        self.action_space = self.vec_envs.action_space

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.lr_scheduler = lr_scheduler
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage

        self.optimizer = optim.RMSprop(
            self.policy.parameters(),
            lr,
            eps=eps,
            alpha=alpha,
        )

        self.reporters = reporters or ()
        self.critic_loss_func = critic_loss_func
        self.log_interval = log_interval
        self.n_steps = n_steps
        self.mini_batch = self.n_steps * self.vec_envs.num_envs

        self._num_frames = 0
        self._fitness_logger = _FitnessLogger(self.vec_envs.num_envs)

    def _train(
        self,
        num_frames: Optional[int] = None,
        stop_time: Optional[int] = None,
    ):
        start_time = time()
        iter_ = count()

        last_state = torch.tensor(self.vec_envs.reset(), dtype=torch.float32, device=self.device)
        last_mask = torch.ones(self.vec_envs.num_envs, dtype=torch.float32, device=self.device)

        for _ in iter_:
            if stop_time and (time() - start_time) >= stop_time:
                break

            if self._num_frames and (self._num_frames >= num_frames):
                break

            last_state, last_mask = self._update(last_state, last_mask)

    def _update(self, last_state: Tensor, last_mask: Tensor, ) -> Tuple[Tensor, Tensor]:
        advantage, entropy, last_mask, last_state, log_probs, returns, values = self._rollout(
            last_mask, last_state)

        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = self.critic_loss_func(returns, values)

        loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        fitness = self._fitness_logger.get_logs_and_reset_buffer()
        self._call_reporters(
            'on_update_end',
            num_frames=self._num_frames,
            fitness=fitness,
        )

        return last_state, last_mask

    def _rollout(self, last_mask, last_state):
        entropy = 0.0
        log_probs = torch.empty(
            (self.n_steps, self.vec_envs.num_envs), dtype=_FLOAT, device=self.device
        )
        values = torch.empty_like(log_probs)
        rewards = torch.empty_like(log_probs)
        masks = torch.empty_like(log_probs)
        for step in range(self.n_steps):
            action, critic_values, action_log_probs, dist_entropy = self.policy(last_state)
            action = action.detach().cpu().numpy()

            # Clip the actions to avoid out of bound error
            clipped_action = action
            if isinstance(self.action_space, Box):
                clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
            else:
                clipped_action = clipped_action.flatten()

            state, reward, done, infos = self.vec_envs.step(clipped_action)
            entropy += dist_entropy.sum()

            log_probs[step] = action_log_probs.flatten()
            values[step] = critic_values.flatten()
            rewards[step] = torch.tensor(reward, dtype=_FLOAT, device=self.device)
            masks[step] = last_mask

            last_mask = torch.tensor(1 - done, device=self.device)
            last_state = torch.tensor(state, dtype=_FLOAT, device=self.device)

            self._num_frames += self.vec_envs.num_envs
            self._fitness_logger.log(reward, done)

        entropy /= self.n_steps * self.vec_envs.num_envs
        _, critic_values, *_ = self.policy(last_state)
        returns, advantage = self._compute_returns_and_advantage(
            critic_values, last_mask, masks, rewards, values
        )
        return advantage, entropy, last_mask, last_state, log_probs, returns, values

    def _compute_returns_and_advantage(
        self, last_values: torch.Tensor, last_mask, masks, rewards, values
    ):
        last_values = last_values.clone().flatten()

        last_gae_lam = 0
        advantages = torch.zeros_like(rewards)
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = last_mask
                next_values = last_values
            else:
                next_non_terminal = masks[step + 1]
                next_values = values[step + 1]
            delta = rewards[step] + self.gamma * next_values * next_non_terminal - values[step]
            last_gae_lam = delta + self.gamma * next_non_terminal * last_gae_lam

            advantages[step] = last_gae_lam

        returns = advantages + values
        return returns, advantages

    def _call_reporters(self, stage: str, *args, **kwargs):
        for reporter in self.reporters:
            getattr(reporter, stage)(*args, **kwargs)


class _FitnessLogger:
    def __init__(self, num_envs: int):
        self._fitness_scores = [0] * num_envs
        self._fitnesses = []
        self._fitness = 0.

    def log(self, reward, done):
        for i, (r, d) in enumerate(zip(reward, done)):
            if not d:
                self._fitness_scores[i] += r
            else:
                self._fitnesses.append(self._fitness_scores[i])
                self._fitness_scores[i] = 0.0

    def get_logs_and_reset_buffer(self):
        if self._fitnesses:
            self._fitness = np.mean(self._fitnesses)
        self._fitnesses = []
        return self._fitness
