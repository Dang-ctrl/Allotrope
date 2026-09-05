"""Score the forecasters against a real simulated year, and print the numbers.

Run with `python -m allotrope.intelligence.forecasting.run_evaluation`. Every
number it prints comes from `allotrope.sim.runner.build_plant` and
`run_episode` executed in this process -- nothing here is a stored or
hand-typed result. See `docs/forecasting.md` for the numbers from the run
that document was written against, and re-run this script to check them.

The plant is driven by `LegacyNPlusOne` (this project's incumbent-practice
baseline) purely to produce a full year of the four signals the environment
observes; the forecasters below never see the controller's actions, only the
resulting demand/availability series, so the choice of controller does not
leak anything into the forecast.
"""

from __future__ import annotations

import numpy as np

from allotrope.control.baseline import LegacyNPlusOne
from allotrope.intelligence.forecasting.evaluation import evaluate_forecaster
from allotrope.intelligence.forecasting.forecasters import (
    EWMAForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)
from allotrope.sim.runner import build_plant, run_episode

SIGNALS = ["electrical_load_kw", "firm_thermal_kw", "pv_available_kw", "wind_available_kw"]
HORIZONS = [1, 24]  # 1-hour-ahead and 24-hour-ahead, at the plant's 1h step
DIURNAL_PERIOD = 24


def collect_signals(seed: int = 0, periods: int = 8760) -> dict[str, np.ndarray]:
    """A full simulated year of the four observed signals, from a real run."""
    plant = build_plant("maitri", periods=periods, seed=seed)
    controller = LegacyNPlusOne(plant.cfg)
    result = run_episode(plant, controller)
    return {signal: result.telemetry[signal].to_numpy() for signal in SIGNALS}


def run(seed: int = 0, periods: int = 8760) -> list[dict]:
    """Evaluate every forecaster x horizon x signal combination; return rows."""
    signals = collect_signals(seed=seed, periods=periods)
    forecasters = [
        PersistenceForecaster(),
        SeasonalNaiveForecaster(period=DIURNAL_PERIOD),
        EWMAForecaster(alpha=0.3),
    ]

    rows: list[dict] = []
    for signal_name, series in signals.items():
        for horizon in HORIZONS:
            for forecaster in forecasters:
                metrics = evaluate_forecaster(
                    series, forecaster, horizon, min_history=DIURNAL_PERIOD
                )
                row = metrics.to_dict()
                row["signal"] = signal_name
                rows.append(row)
    return rows


def _print_table(rows: list[dict]) -> None:
    header = f"{'signal':<22}{'horizon':>8}{'forecaster':>16}{'MAE':>12}{'RMSE':>12}{'MAPE %':>10}{'n':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        mape = f"{row['mape_pct']:.2f}" if row["mape_pct"] is not None else "n/a"
        print(
            f"{row['signal']:<22}{row['horizon']:>8}{row['name']:>16}"
            f"{row['mae']:>12.3f}{row['rmse']:>12.3f}{mape:>10}{row['n']:>8}"
        )


if __name__ == "__main__":
    _print_table(run())
