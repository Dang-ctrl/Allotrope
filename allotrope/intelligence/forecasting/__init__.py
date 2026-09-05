"""Standalone forecasting of the plant's observed demand/availability signals.

Forecasts `electrical_load_kw`, `firm_thermal_kw`, `pv_available_kw` and
`wind_available_kw` -- the four series `PolarMicrogridEnv._observe` reads
from `plant.observe()` -- at 1-step and 24-step (1h/24h, at this project's
1h dispatch interval) horizons, using only strictly-past data.

Not yet consumed by `allotrope.agents`, `allotrope.envs` or
`allotrope.safety` -- see `docs/forecasting.md`.
"""

from allotrope.intelligence.forecasting.evaluation import (
    ForecastMetrics,
    compute_metrics,
    evaluate_forecaster,
)
from allotrope.intelligence.forecasting.forecasters import (
    EWMAForecaster,
    Forecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)

__all__ = [
    "Forecaster",
    "PersistenceForecaster",
    "SeasonalNaiveForecaster",
    "EWMAForecaster",
    "ForecastMetrics",
    "compute_metrics",
    "evaluate_forecaster",
]
