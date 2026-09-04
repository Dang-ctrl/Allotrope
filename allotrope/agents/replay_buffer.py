"""A fixed-size replay buffer holding joint transitions.

One buffer, shared by both agents. The environment's action is a single Dict
(`genset_on` and `dispatch` chosen together and scored by one reward), so a
transition is stored once and each agent reads out only the slice of the
action it is responsible for. This keeps DQN and SDDPG training against
exactly the same experience -- including the consequence of the *other*
agent's half of the action -- rather than against two separate, inconsistent
histories.

Plain numpy ring buffer, not a library, because the state here is small
enough (a few thousand transitions of a ~20-float observation) that nothing
beyond contiguous arrays and random-index sampling is warranted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    genset_on: np.ndarray
    dispatch: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, n_gensets: int, dispatch_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.genset_on = np.zeros((capacity, n_gensets), dtype=np.float32)
        self.dispatch = np.zeros((capacity, dispatch_dim), dtype=np.float32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, t: Transition) -> None:
        i = self._ptr
        self.obs[i] = t.obs
        self.next_obs[i] = t.next_obs
        self.genset_on[i] = t.genset_on
        self.dispatch[i] = t.dispatch
        self.reward[i] = t.reward
        self.done[i] = float(t.done)
        self._ptr = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        idx = rng.integers(0, self._size, size=min(batch_size, self._size))
        return {
            "obs": self.obs[idx],
            "next_obs": self.next_obs[idx],
            "genset_on": self.genset_on[idx],
            "dispatch": self.dispatch[idx],
            "reward": self.reward[idx],
            "done": self.done[idx],
        }


__all__ = ["ReplayBuffer", "Transition"]
