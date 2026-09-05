"""Leakage-safe rolling evaluation of forecasters against a real signal.

The one rule that makes any of these numbers trustworthy: a forecaster
predicting the value at time `t + horizon` must never be given anything from
time `t + 1` onward. `evaluate_forecaster` enforces this structurally, not by
convention -- at each step `t` it hands the forecaster `series[: t + 1]`
(everything through `t`, nothing after), so there is no array to slice wrong.
`tests/test_forecasting.py::test_no_leakage` further checks this by mutating
the *future* tail of the series after generating a forecast and confirming
the forecast is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allotrope.intelligence.forecasting.forecasters import Forecaster


@dataclass(frozen=True)
class ForecastMetrics:
    """MAE, RMSE and (where defined) MAPE for one forecaster/horizon/signal."""

    name: str
    horizon: int
    n: int
    mae: float
    rmse: float
    mape_pct: float | None  # None when the signal crosses (or sits at) zero

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "name": self.name,
            "horizon": self.horizon,
            "n": self.n,
            "mae": self.mae,
            "rmse": self.rmse,
            "mape_pct": self.mape_pct,
        }


def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    name: str = "",
    horizon: int = 0,
    mape_zero_atol: float = 1e-6,
) -> ForecastMetrics:
    """MAE, RMSE, and MAPE restricted to points where `actual` is away from zero.

    PV and wind availability are legitimately zero for long stretches (polar
    night, a calm). Dividing by zero there would either crash or silently
    produce `inf`/`nan` that then poisons a mean; excluding those points and
    reporting how many were excluded (`n`) is the honest alternative to
    either.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")
    if actual.size == 0:
        raise ValueError("cannot compute metrics on an empty series")

    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))

    nonzero = np.abs(actual) > mape_zero_atol
    if np.any(nonzero):
        mape_pct = float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100.0)
    else:
        mape_pct = None

    return ForecastMetrics(
        name=name, horizon=horizon, n=int(actual.size), mae=mae, rmse=rmse, mape_pct=mape_pct
    )


def evaluate_forecaster(
    series: np.ndarray,
    forecaster: Forecaster,
    horizon: int,
    min_history: int = 1,
) -> ForecastMetrics:
    """Roll `forecaster` forward over `series`, scoring `horizon`-step-ahead predictions.

    At each candidate origin `t` (0-indexed), the forecaster is given
    `series[: t + 1]` -- strictly the past -- and asked to predict
    `series[t + horizon]`. `min_history` skips origins where the forecaster
    would be asked to predict from fewer observations than it needs to be
    meaningful (a `SeasonalNaiveForecaster` still runs before that, via its
    own persistence fallback, but comparing it before it has seen a full
    cycle is not an interesting number).
    """
    series = np.asarray(series, dtype=float)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    n = len(series)
    if n <= horizon:
        raise ValueError("series must be longer than the horizon")

    start = max(min_history - 1, 0)
    stop = n - horizon
    if start >= stop:
        raise ValueError("not enough data for any origin at this horizon/min_history")

    predicted = np.empty(stop - start)
    actual = np.empty(stop - start)
    for i, t in enumerate(range(start, stop)):
        history = series[: t + 1]
        predicted[i] = forecaster.forecast(history, horizon)
        actual[i] = series[t + horizon]

    return compute_metrics(actual, predicted, name=forecaster.name, horizon=horizon)


__all__ = ["ForecastMetrics", "compute_metrics", "evaluate_forecaster"]
