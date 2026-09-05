"""Projects the typed station config dataclasses to JSON for the API layer.

Kept separate from `allotrope.config` itself: that module's job is to load and
validate a station; this module's job is to decide what a browser is allowed
to see of it.
"""

from __future__ import annotations

from allotrope.config import StationConfig


def station_to_dict(cfg: StationConfig) -> dict:
    return {
        "id": cfg.site.id,
        "name": cfg.site.name,
        "lat": cfg.site.latitude_deg,
        "lon": cfg.site.longitude_deg,
        "elevation_m": cfg.site.elevation_m,
        "is_polar": cfg.site.is_polar,
        "occupancy": {
            "winter_crew": cfg.occupancy.winter_crew,
            "summer_crew": cfg.occupancy.summer_crew,
        },
        "gensets": [
            {
                "id": g.id,
                "rated_kw": g.rated_kw,
                "chp_heat_ratio": g.chp_heat_ratio,
                "wet_stack_threshold_frac": g.wet_stack_threshold_frac,
                "burn_off_threshold_frac": g.burn_off_threshold_frac,
                "min_stable_load_frac": g.min_stable_load_frac,
            }
            for g in cfg.gensets
        ],
        "storage": [
            {
                "id": s.id,
                "chemistry": s.chemistry,
                "location": s.location,
                "capacity_kwh": s.capacity_kwh,
                "soc_min": s.soc_min,
                "soc_max": s.soc_max,
            }
            for s in cfg.storage
        ],
        "criticality": {
            "life_support_kw": cfg.criticality.life_support_kw,
            "min_indoor_temp_c": cfg.criticality.min_indoor_temp_c,
            "reserve_margin_kw": cfg.criticality.reserve_margin_kw,
        },
        "total_genset_kw": cfg.total_genset_kw,
        "total_storage_kwh": cfg.total_storage_kwh,
    }


__all__ = ["station_to_dict"]
