#calculate green time
"""
Green Time Calculator (grounded, combined-demand version)
===========================================================

Given the phase the Decision Engine has already chosen, works out how long
its green interval should run for.

Formula
-------
This is a demand-responsive extension of the standard queue-clearance /
saturation-flow model used in actuated signal control: green time is sized
to the normalised traffic demand pressing on the phase, bounded by the
configured minimum and maximum green.

For each *direction* (e.g. "N", "S", "E", "W"), three normalised demand
components are built, each independently capped to [0, 1] so that no single
component can dominate just because its raw units happen to run higher:

    - density_ratio = min(1.0, density / config.DENSITY_THRESHOLD)
    - wait_ratio     = min(1.0, wait_time / config.MAX_WAIT_SECONDS)
    - queue_ratio    = min(1.0, queue_length / max_clearable_queue)

        where max_clearable_queue is how many vehicles a MAX_GREEN_TIME
        green could clear at the standard saturation flow rate -- this is
        what grounds the queue term in something physical rather than an
        arbitrary vehicle count.

Predicted metrics (from predictor.py) are blended into each raw value
before it's normalised, so a direction with traffic *about to arrive* gets
extra green even if its current reading looks calm.

A phase can serve two directions at once (e.g. "NS" = N and S flowing
together). Per the project specification, the green time must be sized to
the *combined* relevant demand of the phase's directions, not just the
single most urgent one -- for example:

    NS_demand = demand_North + demand_South
    EW_demand = demand_East + demand_West

This is implemented per demand component: each of density_ratio, wait_ratio,
and queue_ratio is summed across every direction in the phase (each
direction's ratio already capped to [0, 1]), and the resulting combined
ratio is itself capped to [0, 1] -- a phase where both directions are
independently saturated should map to full saturation, not to more than
"fully saturated". The three combined ratios are then averaged into a
single phase urgency score:

    phase_urgency = min(1.0, (density_ratio_sum + wait_ratio_sum + queue_ratio_sum) / 3.0)

and linearly mapped onto the green-time range:

    green_time = MIN_GREEN_TIME + phase_urgency * (MAX_GREEN_TIME - MIN_GREEN_TIME)

Emergency vehicles
-------------------
decision_engine.DecisionEngine already short-circuits around this
calculator entirely when an emergency is active -- it takes `hold_time`
directly from emergency_preemption.check() instead of calling calculate().
`emergency_active` is still accepted here (default False) so this
calculator behaves sensibly if ever called directly/standalone.

Interface
---------
    calculator.calculate(phase, current_metrics, predicted_metrics=None,
                          isolated_approaches=None, emergency_active=False) -> float

This still matches what decision_engine.DecisionEngine calls:
    self.green_time_calculator.calculate(phase, current_metrics, predicted_metrics)
"""

from typing import Dict, Optional, Set

import config

# --- Reference value (standard traffic-engineering figure) --------------
# ~1800 vehicles/hour/lane is a commonly used saturation flow rate; convert
# to vehicles/second for per-step math. Used only to ground the queue term
# in a physically meaningful "how many vehicles could a max green clear".
SATURATION_FLOW_VEHICLES_PER_SEC = 1800 / 3600  # = 0.5

# How much predicted_metrics gets trusted vs. current_metrics for each
# blended ingredient. 0 = ignore predictions entirely, 1 = trust them fully.
PREDICTION_BLEND_WEIGHT = 0.3


class GreenTimeCalculator:
    """Computes the green duration for a chosen phase from a grounded urgency average."""

    def __init__(
        self,
        saturation_flow_veh_per_sec: Optional[float] = None,
        prediction_blend_weight: Optional[float] = None,
    ):
        self.saturation_flow = saturation_flow_veh_per_sec or SATURATION_FLOW_VEHICLES_PER_SEC
        self.prediction_blend_weight = (
            PREDICTION_BLEND_WEIGHT if prediction_blend_weight is None else prediction_blend_weight
        )
        # How many vehicles a full MAX_GREEN_TIME green could clear at the
        # saturation flow rate -- the reference used to normalise queue_ratio.
        self.max_clearable_queue = self.saturation_flow * config.MAX_GREEN_TIME

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def calculate(
        self,
        phase: str,
        current_metrics: Dict[str, Dict[str, float]],
        predicted_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        isolated_approaches: Optional[Set[str]] = None,
        emergency_active: bool = False,
    ) -> float:
        """
        Returns the recommended green duration for `phase`, already
        clamped to [config.MIN_GREEN_TIME, config.MAX_GREEN_TIME].
        """
        if phase not in config.PHASES:
            raise ValueError(f"Unknown phase {phase!r}; expected one of {config.PHASES}")

        if emergency_active:
            return config.MAX_GREEN_TIME

        isolated_approaches = isolated_approaches or set()
        directions = [
            d for d in config.PHASE_MAP[phase] if d not in isolated_approaches
        ]

        if not directions:
            # Every approach in this phase is isolated -- nothing is
            # actually being served, so there's nothing to size the green
            # time against. Fall back to the minimum.
            return config.MIN_GREEN_TIME

        # Per the spec, the phase's green time must be sized to the
        # COMBINED demand of its directions (e.g. NS_demand = demand_North
        # + demand_South), not just whichever single direction is most
        # urgent. Sum each normalised demand component across directions,
        # cap each combined component back to [0, 1], then average.
        density_ratio_sum = 0.0
        wait_ratio_sum = 0.0
        queue_ratio_sum = 0.0
        for d in directions:
            density_ratio, wait_ratio, queue_ratio = self._ratios_for_direction(
                d, current_metrics, predicted_metrics
            )
            density_ratio_sum += density_ratio
            wait_ratio_sum += wait_ratio
            queue_ratio_sum += queue_ratio

        density_ratio_sum = min(1.0, density_ratio_sum)
        wait_ratio_sum = min(1.0, wait_ratio_sum)
        queue_ratio_sum = min(1.0, queue_ratio_sum)

        phase_urgency = min(
            1.0, (density_ratio_sum + wait_ratio_sum + queue_ratio_sum) / 3.0
        )

        green_range = config.MAX_GREEN_TIME - config.MIN_GREEN_TIME
        green_time = config.MIN_GREEN_TIME + phase_urgency * green_range
        return self._clamp(green_time)

    # ------------------------------------------------------------------ #
    # Per-direction normalised demand components
    # ------------------------------------------------------------------ #
    def _ratios_for_direction(
        self,
        direction: str,
        current_metrics: Dict[str, Dict[str, float]],
        predicted_metrics: Optional[Dict[str, Dict[str, float]]],
    ):
        """Returns (density_ratio, wait_ratio, queue_ratio) for one direction,
        each already blended with its prediction and capped to [0, 1]."""
        current = current_metrics.get(direction, {})
        predicted = (predicted_metrics or {}).get(direction, {})

        effective_density = self._blend(
            current.get("density", 0.0), predicted.get("density")
        )
        effective_wait = self._blend(
            current.get("wait_time", current.get("waiting_time", 0.0)),
            predicted.get("wait_time", predicted.get("waiting_time", None)),
        )

        effective_queue = self._blend(
            current.get("queue_length", 0.0), predicted.get("queue_length")
        )

        density_ratio = (
            min(effective_density / config.DENSITY_THRESHOLD, 1.0)
            if config.DENSITY_THRESHOLD > 0 else 0.0
        )
        wait_ratio = (
            min(effective_wait / config.MAX_WAIT_SECONDS, 1.0)
            if config.MAX_WAIT_SECONDS > 0 else 0.0
        )
        queue_ratio = (
            min(effective_queue / self.max_clearable_queue, 1.0)
            if self.max_clearable_queue > 0 else 0.0
        )

        return density_ratio, wait_ratio, queue_ratio

    def _blend(self, current_value: float, predicted_value: Optional[float]) -> float:
        """Weighted blend of a current reading and its predicted counterpart."""
        if predicted_value is None:
            return current_value
        w = self.prediction_blend_weight
        return (1 - w) * current_value + w * predicted_value

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clamp(green_time: float) -> float:
        return max(config.MIN_GREEN_TIME, min(config.MAX_GREEN_TIME, green_time))