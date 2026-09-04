"""Federated averaging: the arithmetic, and the end-to-end round.

Maitri and Bharati differ by roughly 2x in installed capacity but share the same
asset counts -- three gensets, two storage packs -- which is what makes
averaging their trained weights meaningful rather than merely mechanically
possible. These tests keep local training tiny (a handful of short episodes)
because the property under test is that the federation mechanism is correct,
not that it converges to a good policy -- that claim belongs to
`scripts/evaluate_agent.py`, on much longer runs.
"""

from __future__ import annotations

import torch

from allotrope.agents.dqn import DQNConfig
from allotrope.agents.federated import (
    FederatedConfig,
    average_state_dicts,
    run_federated_training,
)
from allotrope.agents.hybrid import HybridAgent
from allotrope.agents.sddpg import SDDPGConfig
from allotrope.config import load_station
from allotrope.envs.polar_microgrid import observation_width
from allotrope.safety.fallback import GuardedController
from allotrope.sim.runner import build_plant, run_episode


def _leaves(state: dict, prefix: tuple = ()) -> list[tuple[tuple, torch.Tensor]]:
    """Flatten a HybridAgent state dict to (key-path, tensor) pairs.

    A path element can itself legally contain a dot -- a plain `nn.Module`
    `state_dict()` key like `"net.0.weight"` is one flat dict key, not three
    nested levels -- so paths are kept as tuples of the original dict keys
    rather than joined into a dotted string that would have to be re-split.
    """
    out = []
    for key, value in state.items():
        path = prefix + (key,)
        if isinstance(value, dict):
            out.extend(_leaves(value, path))
        else:
            out.append((path, value))
    return out


def _lookup(state: dict, path: tuple):
    node = state
    for key in path:
        node = node[key]
    return node


def test_averaging_two_identical_networks_returns_the_same_weights():
    cfg = load_station("maitri")
    agent = HybridAgent(cfg)
    state = agent.state_dict()
    averaged = average_state_dicts([state, state])
    for path, tensor in _leaves(state):
        assert torch.allclose(_lookup(averaged, path), tensor)


def test_averaging_is_the_elementwise_mean():
    a = {"net": {"w": torch.tensor([0.0, 2.0]), "b": torch.tensor([1.0])}}
    b = {"net": {"w": torch.tensor([2.0, 4.0]), "b": torch.tensor([3.0])}}
    averaged = average_state_dicts([a, b])
    assert torch.allclose(averaged["net"]["w"], torch.tensor([1.0, 3.0]))
    assert torch.allclose(averaged["net"]["b"], torch.tensor([2.0]))


def test_averaging_respects_unequal_weights():
    a = {"net": {"w": torch.tensor([0.0])}}
    b = {"net": {"w": torch.tensor([10.0])}}
    averaged = average_state_dicts([a, b], weights=[3.0, 1.0])
    assert torch.allclose(averaged["net"]["w"], torch.tensor([2.5]))


def test_averaging_requires_at_least_one_state_dict():
    import pytest

    with pytest.raises(ValueError):
        average_state_dicts([])


def test_maitri_and_bharati_agents_share_dimensions():
    """This is the precondition federation depends on: same shapes to average."""
    maitri, bharati = load_station("maitri"), load_station("bharati")
    assert observation_width(maitri) == observation_width(bharati)
    assert len(maitri.gensets) == len(bharati.gensets)
    assert len(maitri.storage) == len(bharati.storage)

    a, b = HybridAgent(maitri), HybridAgent(bharati)
    assert a.obs_dim == b.obs_dim
    assert a.dispatch_dim == b.dispatch_dim
    assert a.dqn.n_actions == b.dqn.n_actions


def test_federation_across_stations_with_different_asset_counts_is_refused():
    maitri = load_station("maitri")
    mismatched = maitri.__class__(**{**maitri.__dict__, "gensets": maitri.gensets[:2]})

    import pytest

    with pytest.raises(ValueError, match="cannot be federated"):
        run_federated_training({"maitri": maitri, "short": mismatched}, FederatedConfig(rounds=1))


def _tiny_config(rounds: int, local_episodes: int) -> FederatedConfig:
    return FederatedConfig(
        rounds=rounds,
        local_episodes=local_episodes,
        episode_steps=24,
        seed=0,
        dqn_config=DQNConfig(warmup_steps=5, batch_size=8, epsilon_decay=0.9),
        sddpg_config=SDDPGConfig(warmup_steps=5, batch_size=8, exploration_decay=0.9),
    )


def test_a_federated_round_actually_changes_the_global_weights():
    stations = {"maitri": load_station("maitri"), "bharati": load_station("bharati")}
    initial = HybridAgent(stations["maitri"]).state_dict()

    global_agent, logs = run_federated_training(stations, _tiny_config(rounds=2, local_episodes=2))

    assert len(logs) == 2
    assert set(logs[0].per_station_mean_reward) == {"maitri", "bharati"}
    after = global_agent.state_dict()
    initial_leaves = dict(_leaves(initial))
    after_leaves = dict(_leaves(after))
    changed = any(
        not torch.allclose(initial_leaves[path], after_leaves[path]) for path in initial_leaves
    )
    assert changed, "federated training left the global weights untouched"


def test_the_federated_agent_stays_safe_at_every_participating_station():
    """Averaged weights are still an arbitrary network; the guarantee must not
    depend on which station trained them."""
    stations = {"maitri": load_station("maitri"), "bharati": load_station("bharati")}
    global_agent, _ = run_federated_training(stations, _tiny_config(rounds=1, local_episodes=1))

    for name, cfg in stations.items():
        guarded = GuardedController(cfg, agent=global_agent)
        plant = build_plant(cfg, start="2026-06-01", periods=48, seed=1)
        result = run_episode(plant, guarded)
        assert result.summary["critical_unserved_kwh"] == 0.0, name
        assert result.summary["freeze_violation_steps"] == 0.0, name


def test_on_round_callback_fires_once_per_round():
    stations = {"maitri": load_station("maitri"), "bharati": load_station("bharati")}
    seen = []
    run_federated_training(
        stations, _tiny_config(rounds=3, local_episodes=1), on_round=lambda log: seen.append(log.round)
    )
    assert seen == [0, 1, 2]
