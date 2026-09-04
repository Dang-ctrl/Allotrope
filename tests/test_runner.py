"""The episode runner's reporting helpers: compare() and compare_multi().

compare_multi exists because an earlier, untested version of this exact
averaging-and-tabulating logic lived only in scripts/evaluate_agent.py and had
its rows and columns transposed -- caught only because the script crashed
outright. Moving it here, with its own tests, is the actual fix; the
crash-when-wrong was luck, not a safeguard.
"""

from __future__ import annotations

import pytest

from allotrope.config import load_station
from allotrope.control.baseline import EfficientRuleBased, LegacyNPlusOne
from allotrope.sim.runner import build_plant, compare, compare_multi, run_episode

WINTER = "2026-06-01"


@pytest.fixture(scope="module")
def cfg():
    return load_station("maitri")


@pytest.fixture(scope="module")
def two_controller_results(cfg):
    results = []
    for controller_cls in (LegacyNPlusOne, EfficientRuleBased):
        plant = build_plant(cfg, start=WINTER, periods=24 * 7, seed=0)
        results.append(run_episode(plant, controller_cls(cfg)))
    return results


def test_compare_lays_out_metrics_as_rows_and_controllers_as_columns(two_controller_results):
    df = compare(two_controller_results)
    assert set(df.columns) == {"legacy_n_plus_one", "efficient_rule_based"}
    assert "fuel_kl" in df.index


def test_compare_multi_has_the_same_orientation_as_compare(cfg, two_controller_results):
    """The bug this guards against: compare_multi transposed relative to compare."""
    by_label = {r.controller: [r] for r in two_controller_results}
    single_seed_df = compare_multi(by_label)

    assert set(single_seed_df.columns) == set(compare(two_controller_results).columns)
    assert list(single_seed_df.index) == list(compare(two_controller_results, keys=list(single_seed_df.index)).index)


def test_compare_multi_averages_across_seeds(cfg):
    controller_factory = EfficientRuleBased
    results = [
        run_episode(build_plant(cfg, start=WINTER, periods=24 * 3, seed=s), controller_factory(cfg))
        for s in (0, 1, 2)
    ]
    by_label = {"efficient_rule_based": results}
    df = compare_multi(by_label, keys=["fuel_kl"])

    expected_mean = sum(r.summary["fuel_kl"] for r in results) / len(results)
    assert df.loc["fuel_kl", "efficient_rule_based"] == pytest.approx(expected_mean)


def test_compare_multi_selected_metrics_are_addressable_by_name(two_controller_results):
    """This is exactly the access pattern that broke: df[[metric_names]]."""
    by_label = {r.controller: [r] for r in two_controller_results}
    df = compare_multi(by_label, keys=["fuel_kl", "genset_starts"])
    subset = df.loc[["fuel_kl", "genset_starts"]]
    assert subset.shape == (2, 2)


def test_compare_multi_rejects_a_label_with_no_results():
    with pytest.raises(ValueError, match="no results"):
        compare_multi({"empty": []})


def test_compare_multi_default_keys_are_all_present_in_a_real_summary(two_controller_results):
    df = compare_multi({r.controller: [r] for r in two_controller_results})
    summary = two_controller_results[0].summary
    for key in df.index:
        assert key in summary
