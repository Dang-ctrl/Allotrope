"""Controllers that dispatch the plant.

Three kinds live under this package, and keeping them behind one interface is
deliberate:

  * `baseline.LegacyNPlusOne` -- current practice at a polar station, and the
    thing this project claims to improve on. Every headline number is a
    comparison against it, so it is written to be a fair opponent rather than a
    strawman.
  * `baseline.EfficientRuleBased` -- what careful engineering achieves without
    any learning at all. A learned policy that cannot beat this is not worth
    deploying, and this is also the fallback the safety layer falls back *to*.
  * the learned agents, added later, which must clear both.
"""

from __future__ import annotations

from typing import Protocol

from allotrope.sim.plant import DispatchCommand, PolarMicrogrid


class Controller(Protocol):
    """Anything that can turn a plant observation into a dispatch command."""

    name: str

    def reset(self) -> None: ...

    def act(self, observation: dict, plant: PolarMicrogrid) -> DispatchCommand: ...


__all__ = ["Controller"]
