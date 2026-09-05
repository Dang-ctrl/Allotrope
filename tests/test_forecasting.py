"""Tests for allotrope.intelligence.forecasting.

Covers: (a) that a forecaster/evaluator structurally cannot see future data,
(b) correctness of the seasonal-naive forecaster on a perfectly periodic
signal, and (c) the metrics computation on a hand-worked example.
"""

from __future__ import annotations

import numpy as np
import pytest

from allotrope.intelligence.forecasting import (
    EWMAForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
    compute_metrics,
    evaluate_forecaster,
)


# -- (a) leakage -------------------------------------------------------------


def test_no_leakage_mutating_future_does_not_change_forecast():
    """A forecast made from history through time t must not depend on t+1..n.

    Generates the same random series twice, mutates everything strictly
    after the origin in one copy, and checks every forecaster produces an
    identical forecast from the (identical) past.
    """
    rng = np.random.default_rng(7)
    base = rng.normal(100.0, 10.0, size=200)
    origin = 150  # forecast is made using base[: origin + 1]

    mutated = base.copy()
    mutated[origin + 1 :] = rng.normal(-999.0, 500.0, size=len(base) - origin - 1)

    forecasters = [
        PersistenceForecaster(),
        SeasonalNaiveForecaster(period=24),
        EWMAForecaster(alpha=0.3),
    ]
    for forecaster in forecasters:
        pred_base = forecaster.forecast(base[: origin + 1], horizon=24)
        pred_mutated = forecaster.forecast(mutated[: origin + 1], horizon=24)
        assert pred_base == pred_mutated, forecaster.name


def test_evaluate_forecaster_never_hands_forecaster_the_future():
    """evaluate_forecaster's own slicing is leak-free: patch forecast() to check."""
    rng = np.random.default_rng(3)
    series = rng.normal(50.0, 5.0, size=100)
    horizon = 24
    seen_lengths = []

    class RecordingForecaster:
        name = "recorder"

        def forecast(self, history, horizon):
            seen_lengths.append(len(history))
            # The last element of history must never equal a value that sits
            # at or after the target index -- check index bounds directly.
            return float(history[-1])

    evaluate_forecaster(series, RecordingForecaster(), horizon, min_history=1)
    # For origin t, evaluate_forecaster must pass exactly t+1 points (indices
    # 0..t), i.e. history never includes index t+horizon or anything after it.
    n = len(series)
    expected = list(range(1, n - horizon + 1))
    assert seen_lengths == expected


def test_persistence_forecaster_uses_only_last_observation():
    history = np.array([1.0, 2.0, 3.0, 999.0])
    forecaster = PersistenceForecaster()
    assert forecaster.forecast(history, horizon=1) == 999.0
    assert forecaster.forecast(history, horizon=24) == 999.0  # ignores horizon entirely


# -- (b) correctness on a known periodic signal ------------------------------


def test_seasonal_naive_recovers_perfectly_periodic_signal():
    """A pure period-24 sine wave should be forecast near-exactly once one full cycle has elapsed."""
    period = 24
    t = np.arange(500)
    series = 10.0 + 5.0 * np.sin(2.0 * np.pi * t / period)

    forecaster = SeasonalNaiveForecaster(period=period)
    metrics = evaluate_forecaster(series, forecaster, horizon=1, min_history=period)
    assert metrics.mae < 1e-9
    assert metrics.rmse < 1e-9

    metrics_24h = evaluate_forecaster(series, forecaster, horizon=24, min_history=period)
    assert metrics_24h.mae < 1e-9


def test_seasonal_naive_falls_back_to_persistence_before_one_period():
    """With less than a full period of history, seasonal-naive must not fabricate data."""
    series = np.array([5.0, 6.0, 7.0])
    forecaster = SeasonalNaiveForecaster(period=24)
    # idx = n-1+horizon-period is negative here -> falls back to history[-1]
    assert forecaster.forecast(series, horizon=1) == 7.0


def test_seasonal_naive_beats_persistence_on_periodic_signal_with_drift():
    """On a signal with real periodic structure, seasonal-naive should out-MAE persistence."""
    period = 24
    t = np.arange(1000)
    rng = np.random.default_rng(1)
    series = 10.0 + 5.0 * np.sin(2.0 * np.pi * t / period) + rng.normal(0.0, 0.05, size=len(t))

    seasonal = evaluate_forecaster(series, SeasonalNaiveForecaster(period=period), horizon=1)
    naive = evaluate_forecaster(series, PersistenceForecaster(), horizon=1)
    assert seasonal.mae < naive.mae


# -- (c) metrics on a hand-worked example ------------------------------------


def test_compute_metrics_hand_worked_example():
    actual = np.array([10.0, 20.0, 0.0, 40.0])
    predicted = np.array([12.0, 18.0, 5.0, 44.0])
    # errors = pred - actual = [2, -2, 5, 4]
    # MAE = (2+2+5+4)/4 = 3.25
    # RMSE = sqrt((4+4+25+16)/4) = sqrt(49/4) = 3.5
    # MAPE: excludes the zero-actual point -> |2/10|, |-2/20|, |4/40| = .2, .1, .1 -> mean *100 = 13.333...
    metrics = compute_metrics(actual, predicted, name="hand", horizon=1)
    assert metrics.mae == pytest.approx(3.25)
    assert metrics.rmse == pytest.approx(3.5)
    assert metrics.mape_pct == pytest.approx(100.0 * (0.2 + 0.1 + 0.1) / 3.0)
    assert metrics.n == 4


def test_compute_metrics_all_zero_actual_gives_none_mape():
    actual = np.zeros(5)
    predicted = np.array([1.0, -1.0, 0.5, 0.0, 2.0])
    metrics = compute_metrics(actual, predicted)
    assert metrics.mape_pct is None
    assert metrics.mae == pytest.approx(np.mean(np.abs(predicted)))


def test_compute_metrics_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        compute_metrics(np.array([1.0, 2.0]), np.array([1.0]))


# -- evaluate_forecaster input validation ------------------------------------


def test_evaluate_forecaster_rejects_series_too_short_for_horizon():
    with pytest.raises(ValueError):
        evaluate_forecaster(np.array([1.0, 2.0]), PersistenceForecaster(), horizon=5)


def test_ewma_forecaster_smooths_toward_recent_level():
    # A step function: EWMA forecast right after the step should sit strictly
    # between the old and new level, unlike persistence (== new level exactly).
    series = np.concatenate([np.zeros(20), np.full(5, 10.0)])
    forecaster = EWMAForecaster(alpha=0.3)
    pred = forecaster.forecast(series, horizon=1)
    assert 0.0 < pred < 10.0
