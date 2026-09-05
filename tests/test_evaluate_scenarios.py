"""Scenario-suite tests: real statistics over real (short, cheap) runs.

Periods are kept short (a few days) and seed counts small so the suite stays
fast, but every run underneath is the same `run_episode` the rest of the
project uses -- no shortcuts, no synthetic fixtures standing in for it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from allotrope.control.baseline import EfficientRuleBased, LegacyNPlusOne
from allotrope.evaluate_scenarios import METRIC_KEYS, run_scenario_suite, summarize


def test_summarize_matches_numpy_reference():
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    stats = summarize(values)
    arr = np.asarray(values)
    assert stats["mean"] == pytest.approx(float(np.mean(arr)))
    assert stats["median"] == pytest.approx(float(np.median(arr)))
    assert stats["std"] == pytest.approx(float(np.std(arr)))
    assert stats["min"] == 1.0
    assert stats["max"] == 100.0


def test_suite_runs_every_seed_and_every_metric():
    seeds = [0, 1, 2]
    result = run_scenario_suite(
        "maitri", EfficientRuleBased, "efficient_rule_based", seeds, periods=48
    )
    assert result.n_seeds == 3
    assert set(result.per_seed) == set(METRIC_KEYS)
    for key in METRIC_KEYS:
        assert len(result.per_seed[key]) == 3
        assert set(result.stats[key]) == {"mean", "median", "std", "min", "max", "p5", "p95"}


def test_different_seeds_give_different_weather_and_therefore_different_fuel():
    result = run_scenario_suite("maitri", EfficientRuleBased, "x", [0, 1, 2, 3, 4], periods=72)
    fuel = result.per_seed["fuel_kl"]
    assert len(set(fuel)) > 1, "independent seeds should not all realise identical fuel use"


def test_legacy_uses_more_fuel_and_starts_fewer_gensets_than_efficient_on_average():
    seeds = [0, 1, 2, 3, 4]
    legacy = run_scenario_suite("maitri", LegacyNPlusOne, "legacy", seeds, periods=240)
    efficient = run_scenario_suite("maitri", EfficientRuleBased, "efficient", seeds, periods=240)
    assert legacy.stats["fuel_kl"]["mean"] > efficient.stats["fuel_kl"]["mean"]
    assert legacy.stats["genset_starts"]["mean"] < efficient.stats["genset_starts"]["mean"]


def test_result_round_trips_through_json():
    result = run_scenario_suite("maitri", EfficientRuleBased, "efficient", [0, 1], periods=48)
    encoded = json.dumps(result.as_dict())
    decoded = json.loads(encoded)
    assert decoded["controller"] == "efficient"
    assert decoded["n_seeds"] == 2
    assert decoded["stats"]["fuel_kl"]["mean"] == pytest.approx(result.stats["fuel_kl"]["mean"])


def test_no_critical_load_ever_lost_by_either_rule_based_baseline():
    seeds = [0, 1, 2, 3, 4, 5]
    for controller_cls, name in [(LegacyNPlusOne, "legacy"), (EfficientRuleBased, "efficient")]:
        result = run_scenario_suite("maitri", controller_cls, name, seeds, periods=168)
        for value in result.per_seed["critical_unserved_kwh"]:
            assert value == pytest.approx(0.0, abs=1e-6), name
