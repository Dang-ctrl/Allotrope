"""Asset health tracking against a real simulated episode.

Runs a short episode with a real controller (`EfficientRuleBased`) against a
real `PolarMicrogrid`, feeds the resulting telemetry through
`AssetHealthTracker`, and checks the accumulated numbers both for internal
sanity and against the plant's own independently-computed summary numbers.
"""

from __future__ import annotations

import math

from allotrope.control.baseline import EfficientRuleBased
from allotrope.intelligence.asset_health import AssetHealthTracker, MetricLabel
from allotrope.sim.runner import build_plant


def _run_short_episode(periods: int = 240):
    plant = build_plant(station="maitri", periods=periods, seed=7)
    controller = EfficientRuleBased(cfg=plant.cfg)
    plant.reset()
    tracker = AssetHealthTracker(plant.cfg, dt_h=plant.dt_h)

    for _ in range(periods):
        observation = plant.observe()
        command = controller.act(observation, plant)
        telemetry = plant.step(command)
        tracker.update(telemetry)

    tracker.reconcile_cold_charge_blocks(plant)
    return plant, tracker


def test_genset_starts_match_plant_summary():
    """The tracker's per-genset starts must sum to the plant's own genset_starts."""
    plant, tracker = _run_short_episode()
    summary = plant.summary()

    tracked_total_starts = sum(g.starts for g in tracker.gensets.values())
    assert tracked_total_starts == summary["genset_starts"]

    for genset, health in zip(plant.gensets, tracker.gensets.values()):
        assert health.starts == genset.state.total_starts


def test_genset_run_hours_match_plant_state():
    """Tracked run hours must match each genset's own `run_hours` state exactly."""
    plant, tracker = _run_short_episode()
    for genset in plant.gensets:
        health = tracker.gensets[genset.id]
        assert math.isclose(health.run_hours, genset.state.run_hours, rel_tol=1e-9, abs_tol=1e-9)


def test_genset_deposit_matches_final_plant_state():
    """The tracker's last-seen deposit must equal the genset's live deposit state."""
    plant, tracker = _run_short_episode()
    for genset in plant.gensets:
        health = tracker.gensets[genset.id]
        assert math.isclose(health.deposit, genset.state.deposit, rel_tol=1e-9, abs_tol=1e-9)


def test_low_load_hours_are_bounded_by_run_hours():
    """Hours below the wet-stacking threshold cannot exceed total run hours."""
    _, tracker = _run_short_episode()
    for health in tracker.gensets.values():
        assert 0.0 <= health.low_load_hours <= health.run_hours + 1e-9


def test_wear_score_is_nonnegative_and_zero_only_with_no_stress():
    """Wear score is a nonnegative combination of starts and deposit, never both zero unused."""
    _, tracker = _run_short_episode()
    for health in tracker.gensets.values():
        metric = health.wear_score()
        assert metric.label is MetricLabel.PROXY
        assert metric.value >= 0.0
        if health.starts == 0 and health.deposit == 0.0:
            assert metric.value == 0.0


def test_battery_full_equivalent_cycles_matches_hand_computation():
    """FEC must equal throughput / (2 * capacity), computed independently here."""
    plant, tracker = _run_short_episode()
    for battery in plant.batteries:
        health = tracker.batteries[battery.id]
        expected_fec = battery.state.throughput_kwh / (2.0 * battery.spec.capacity_kwh)
        metric = health.full_equivalent_cycles()
        assert metric.label is MetricLabel.ESTIMATED
        assert math.isclose(metric.value, expected_fec, rel_tol=1e-9, abs_tol=1e-12)
        # And the tracker's own throughput accumulation must match the plant's.
        assert math.isclose(health.throughput_kwh, battery.state.throughput_kwh, rel_tol=1e-9)


def test_battery_soc_extreme_hours_are_bounded():
    """Time spent near either SOC bound cannot exceed the episode length."""
    plant, tracker = _run_short_episode(periods=240)
    episode_hours = 240 * plant.dt_h
    for health in tracker.batteries.values():
        assert 0.0 <= health.low_soc_hours <= episode_hours + 1e-9
        assert 0.0 <= health.high_soc_hours <= episode_hours + 1e-9


def test_cold_charge_blocks_reconciled_from_plant():
    """Reconciling against the live plant must match its own cumulative count."""
    plant, tracker = _run_short_episode()
    for battery in plant.batteries:
        health = tracker.batteries[battery.id]
        assert health.cold_charge_blocks == battery.state.cold_charge_blocks


def test_report_labels_every_metric():
    """Every single exposed field must carry one of the four provenance labels."""
    _, tracker = _run_short_episode()
    report = tracker.report()
    valid_labels = {label.value for label in MetricLabel}
    for section in ("gensets", "batteries"):
        for _, metrics in report[section].items():
            for _, metric in metrics.items():
                assert metric["label"] in valid_labels
                assert "value" in metric
