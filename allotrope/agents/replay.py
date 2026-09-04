"""A replay buffer shared by both learners.

DQN and SDDPG train from the same experience stream -- one environment step
produces one discrete transition and one continuous transition, since the two
agents act on the same timestep -- so one buffer implementation serves both. It
stores plain numpy arrays rather than tensors, which keeps it framework-agnostic
and cheap to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    """A sampled minibatch of transitions, already stacked into arrays."""

    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_obs: np.ndarray
    done: np.ndarray


class ReplayBuffer:
    """A fixed-capacity ring buffer of (obs, action, reward, next_obs, done).

    Capacity is bounded because a year of hourly transitions is under 9 000
    steps -- small enough that the whole training run fits in memory many times
    over, so there is no pressure to do anything cleverer than a ring buffer.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self._size = 0
        self._cursor = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._cursor
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self._cursor = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        if self._size < batch_size:
            raise ValueError(
                f"buffer holds {self._size} transitions, cannot sample {batch_size}"
            )
        idx = rng.integers(0, self._size, size=batch_size)
        return Batch(
            obs=self.obs[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            next_obs=self.next_obs[idx],
            done=self.done[idx],
        )


__all__ = ["ReplayBuffer", "Batch"]
