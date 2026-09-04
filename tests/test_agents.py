"""The learners: mechanics, not learning curves.

These tests do not assert that training makes the agent good -- that is what
`scripts/evaluate_agent.py` is for, on held-out seeds, reported honestly whether
or not the number is flattering. What belongs in the test suite is narrower and
more durable: the replay buffer stores what it is given, both networks produce
actions inside their declared bounds under any input including untrained random
weights, gradients actually flow, checkpoints round-trip, and -- the one
property this project cannot compromise on -- an agent behind the safety layer
cannot endanger the station even before it has learned anything at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from allotrope.agents.checkpoint import load, save
from allotrope.agents.dqn import DQNAgent, DQNConfig, enumerate_commitments
from allotrope.agents.hybrid import HybridAgent
from allotrope.agents.replay import ReplayBuffer
from allotrope.agents.sddpg import SDDPGAgent, SDDPGConfig
from allotrope.agents.train import TrainConfig, make_training_env, train
from allotrope.config import load_station
from allotrope.envs.polar_microgrid import PolarMicrogridEnv, observation_width
from allotrope.safety.fallback import GuardedController
from allotrope.sim.runner import build_plant, run_episode

WINTER = "2026-06-01"


@pytest.fixture(scope="module")
def cfg():
    return load_station("maitri")


@pytest.fixture
def obs_dim(cfg):
    return observation_width(cfg)


# -- replay buffer --------------------------------------------------------


def test_buffer_stores_and_returns_what_it_was_given():
    buf = ReplayBuffer(capacity=10, obs_dim=3, action_dim=2)
    obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    action = np.array([0.5, -0.5], dtype=np.float32)
    buf.add(obs, action, reward=1.5, next_obs=obs * 2, done=False)

    assert len(buf) == 1
    batch = buf.sample(1, np.random.default_rng(0))
    assert np.array_equal(batch.obs[0], obs)
    assert np.array_equal(batch.action[0], action)
    assert batch.reward[0] == pytest.approx(1.5)


def test_buffer_wraps_at_capacity():
    buf = ReplayBuffer(capacity=5, obs_dim=1, action_dim=1)
    for i in range(8):
        buf.add(np.array([float(i)]), np.array([0.0]), 0.0, np.array([0.0]), False)
    assert len(buf) == 5
    # Only the most recent 5 values should remain: 3, 4, 5, 6, 7.
    assert set(buf.obs[:, 0].tolist()) == {3.0, 4.0, 5.0, 6.0, 7.0}


def test_sampling_more_than_available_raises():
    buf = ReplayBuffer(capacity=10, obs_dim=1, action_dim=1)
    buf.add(np.array([0.0]), np.array([0.0]), 0.0, np.array([0.0]), False)
    with pytest.raises(ValueError):
        buf.sample(5, np.random.default_rng(0))


# -- DQN --------------------------------------------------------------------


def test_commitments_are_enumerated_exhaustively_and_reproducibly():
    a = enumerate_commitments(3)
    b = enumerate_commitments(3)
    assert a == b
    assert len(a) == 8
    assert len(set(a)) == 8
    assert (False, False, False) in a
    assert (True, True, True) in a


def test_dqn_action_is_always_a_valid_commitment(obs_dim):
    agent = DQNAgent(obs_dim, n_gensets=3, config=DQNConfig(warmup_steps=0))
    obs = np.zeros(obs_dim, dtype=np.float32)
    rng = np.random.default_rng(0)
    for _ in range(20):
        index, commitment = agent.act(obs, explore=True, rng=rng)
        assert 0 <= index < 8
        assert commitment in enumerate_commitments(3)


def test_dqn_greedy_action_is_deterministic(obs_dim):
    agent = DQNAgent(obs_dim, n_gensets=3, config=DQNConfig(warmup_steps=0))
    obs = np.random.default_rng(1).normal(size=obs_dim).astype(np.float32)
    i1, c1 = agent.act(obs, explore=False)
    i2, c2 = agent.act(obs, explore=False)
    assert i1 == i2 and c1 == c2


def test_dqn_gradients_actually_move_the_weights(obs_dim):
    agent = DQNAgent(obs_dim, n_gensets=3, config=DQNConfig(warmup_steps=0, batch_size=16))
    rng = np.random.default_rng(0)
    for _ in range(64):
        obs = rng.normal(size=obs_dim).astype(np.float32)
        next_obs = rng.normal(size=obs_dim).astype(np.float32)
        agent.observe(obs, int(rng.integers(0, 8)), float(rng.normal()), next_obs, False)

    before = agent.q.net[0].weight.clone()
    stats = agent.update(rng)
    after = agent.q.net[0].weight

    assert stats is not None and np.isfinite(stats["q_loss"])
    assert not torch.equal(before, after)


def test_dqn_state_dict_round_trips(obs_dim):
    agent = DQNAgent(obs_dim, n_gensets=3)
    obs = np.random.default_rng(2).normal(size=obs_dim).astype(np.float32)
    _, before = agent.act(obs, explore=False)

    clone = DQNAgent(obs_dim, n_gensets=3)
    clone.load_state_dict(agent.state_dict())
    _, after = clone.act(obs, explore=False)
    assert before == after


# -- SDDPG --------------------------------------------------------------------


def test_sddpg_action_is_always_inside_the_unit_box(obs_dim):
    agent = SDDPGAgent(obs_dim, action_dim=6, config=SDDPGConfig(warmup_steps=0))
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = rng.normal(size=obs_dim).astype(np.float32) * 10  # exercise clipping too
        action = agent.act(obs, explore=True, rng=rng)
        assert action.shape == (6,)
        assert np.all(action >= -1.0 - 1e-6) and np.all(action <= 1.0 + 1e-6)


def test_sddpg_greedy_action_is_deterministic(obs_dim):
    agent = SDDPGAgent(obs_dim, action_dim=6, config=SDDPGConfig(warmup_steps=0))
    obs = np.random.default_rng(1).normal(size=obs_dim).astype(np.float32)
    a1 = agent.act(obs, explore=False)
    a2 = agent.act(obs, explore=False)
    assert np.array_equal(a1, a2)


def test_sddpg_gradients_actually_move_the_weights(obs_dim):
    agent = SDDPGAgent(obs_dim, action_dim=6, config=SDDPGConfig(warmup_steps=0, batch_size=16))
    rng = np.random.default_rng(0)
    for _ in range(64):
        obs = rng.normal(size=obs_dim).astype(np.float32)
        action = rng.uniform(-1, 1, size=6).astype(np.float32)
        agent.observe(obs, action, float(rng.normal()), rng.normal(size=obs_dim).astype(np.float32), False)

    before = agent.actor.net[0].weight.clone()
    stats = agent.update(rng)
    after = agent.actor.net[0].weight

    assert stats is not None
    assert np.isfinite(stats["critic_loss"]) and np.isfinite(stats["actor_loss"])
    assert not torch.equal(before, after)


def test_sddpg_state_dict_round_trips(obs_dim):
    agent = SDDPGAgent(obs_dim, action_dim=6)
    obs = np.random.default_rng(2).normal(size=obs_dim).astype(np.float32)
    before = agent.act(obs, explore=False)

    clone = SDDPGAgent(obs_dim, action_dim=6)
    clone.load_state_dict(agent.state_dict())
    after = clone.act(obs, explore=False)
    assert np.array_equal(before, after)


def test_exploration_noise_decays_but_not_below_the_floor():
    agent = SDDPGAgent(obs_dim=5, action_dim=2, config=SDDPGConfig(exploration_decay=0.5, exploration_sigma_min=0.05))
    for _ in range(50):
        agent.end_episode()
    assert agent.sigma == pytest.approx(0.05)


# -- HybridAgent --------------------------------------------------------------


def test_hybrid_training_action_matches_the_env_action_space(cfg, obs_dim):
    env = PolarMicrogridEnv(station=cfg, start=WINTER, periods=48, seed=0)
    agent = HybridAgent(cfg)
    obs, _ = env.reset(seed=0)
    action, index = agent.act_training(obs, explore=True, rng=np.random.default_rng(0))
    assert env.action_space.contains(action)
    assert 0 <= index < 2 ** len(cfg.gensets)


def test_hybrid_agent_can_drive_the_environment_for_a_full_episode(cfg):
    env = PolarMicrogridEnv(station=cfg, start=WINTER, periods=48, seed=0, episode_steps=48)
    agent = HybridAgent(cfg)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    while True:
        action, _ = agent.act_training(obs, explore=True, rng=rng)
        obs, _, terminated, truncated, _ = env.step(action)
        assert np.isfinite(obs).all()
        if terminated or truncated:
            break


def test_hybrid_agent_satisfies_the_controller_protocol(cfg):
    """It must be usable exactly like the rule-based baselines: act(obs, plant)."""
    agent = HybridAgent(cfg)
    plant = build_plant(cfg, start=WINTER, periods=24, seed=0)
    result = run_episode(plant, agent)
    assert result.controller == agent.name
    assert np.isfinite(result.summary["fuel_l"])


def test_an_untrained_hybrid_agent_cannot_endanger_the_station_when_guarded(cfg):
    """The safety guarantee must hold even before the agent has learned anything.

    An untrained network's output is close to arbitrary, so this is a real test
    of the projection layer, not a test that happens to pass because the agent
    is already sensible.
    """
    agent = HybridAgent(cfg)
    guarded = GuardedController(cfg, agent=agent)
    plant = build_plant(cfg, start=WINTER, periods=24 * 14, seed=0)
    result = run_episode(plant, guarded)

    assert result.summary["critical_unserved_kwh"] == pytest.approx(0.0, abs=1e-9)
    assert result.summary["freeze_violation_steps"] == 0.0


def test_hybrid_checkpoint_round_trips_through_disk(cfg, tmp_path):
    agent = HybridAgent(cfg)
    path = tmp_path / "agent.pt"
    save(agent, path)
    restored = load(path, cfg)

    plant = build_plant(cfg, start=WINTER, periods=24, seed=0)
    obs1 = plant.observe()
    a1 = agent.act(obs1, plant)
    a2 = restored.act(obs1, plant)
    assert a1.genset_on == a2.genset_on
    assert a1.genset_setpoint_kw == pytest.approx(a2.genset_setpoint_kw, abs=1e-5)
    assert a1.snow_melt_kw == pytest.approx(a2.snow_melt_kw, abs=1e-5)


def test_checkpoint_refuses_to_load_against_a_different_station(cfg, tmp_path):
    agent = HybridAgent(cfg)
    path = tmp_path / "agent.pt"
    save(agent, path)

    other = load_station("bharati")
    with pytest.raises(ValueError, match="cannot load"):
        load(path, other)


# -- the training loop, mechanically -----------------------------------------


def test_training_runs_for_a_few_episodes_without_diverging(cfg):
    train_cfg = TrainConfig(episodes=3, episode_steps=24, update_every=2, seed=0)
    env = make_training_env(cfg, train_cfg, periods=24 * 10)
    agent = HybridAgent(
        cfg,
        dqn_config=DQNConfig(warmup_steps=10, batch_size=8),
        sddpg_config=SDDPGConfig(warmup_steps=10, batch_size=8),
    )

    logs = train(agent, env, train_cfg)

    assert len(logs) == 3
    for log in logs:
        assert np.isfinite(log.reward)
        assert np.isfinite(log.fuel_l)
    # Exploration should have decayed monotonically across episodes.
    assert logs[-1].dqn_epsilon <= logs[0].dqn_epsilon
    assert logs[-1].sddpg_sigma <= logs[0].sddpg_sigma


def test_training_never_lets_life_support_go_unserved(cfg):
    """Training explores behind the same projection layer deployment uses."""
    train_cfg = TrainConfig(episodes=2, episode_steps=24 * 3, update_every=4, seed=0)
    env = make_training_env(cfg, train_cfg, periods=24 * 20)
    agent = HybridAgent(
        cfg,
        dqn_config=DQNConfig(warmup_steps=5, batch_size=8),
        sddpg_config=SDDPGConfig(warmup_steps=5, batch_size=8),
    )

    logs = train(agent, env, train_cfg)
    for log in logs:
        assert log.critical_unserved_kwh == pytest.approx(0.0, abs=1e-9)
        assert log.freeze_violation_steps == 0.0
