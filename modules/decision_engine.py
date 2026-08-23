"""Decision Engine for the Bengaluru adaptive traffic controller.

The Decision Engine coordinates emergency override, road isolation,
starvation protection, QUBO construction/solving, and green-time calculation.
It does not communicate with SUMO directly.  Signal actuation is performed by
main_controller.py through signal_controller.py so the signal state machine can
advance every simulation step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

import config


@dataclass
class Decision:
    """Final decision produced for one junction."""

    phase: str
    green_time: float
    reason: str
    emergency: bool = False
    starvation_override: bool = False
    isolated_approaches: Set[str] = field(default_factory=set)
    candidate_approaches: Set[str] = field(default_factory=set)
    solver_result: Optional[Dict[str, Any]] = None
    qubo: Optional[Dict[str, Any]] = None


class DecisionEngine:
    """Make one validated signal decision from prepared traffic inputs."""

    def __init__(self, qubo_builder, solver_manager, green_time_calculator):
        self.qubo_builder = qubo_builder
        self.solver_manager = solver_manager
        self.green_time_calculator = green_time_calculator

    def decide(
        self,
        current_metrics: Dict[str, Dict[str, float]],
        predicted_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        emergency_result: Optional[Dict[str, Any]] = None,
        isolated_approaches: Optional[Set[str]] = None,
        junction_id: Optional[str] = None,
        solver_name: Optional[str] = None,
    ) -> Decision:
        predicted_metrics = predicted_metrics or {}
        isolated = set(isolated_approaches or set())
        candidates = self._candidate_phases(isolated)

        if emergency_result and emergency_result.get("emergency_active"):
            phase = emergency_result.get("override_phase")
            if phase in config.PHASES and self._phase_available(phase, candidates):
                green = emergency_result.get("green_duration") or config.MAX_GREEN_TIME
                return Decision(
                    phase=phase,
                    green_time=self._validate_green_time(green),
                    reason="emergency_preemption",
                    emergency=True,
                    isolated_approaches=isolated,
                    candidate_approaches=candidates,
                )

        if not candidates:
            return Decision(
                phase=config.PHASES[0],
                green_time=config.MIN_GREEN_TIME,
                reason="all_approaches_isolated_fallback",
                isolated_approaches=isolated,
                candidate_approaches=candidates,
            )

        starved = self._find_starved_phase(current_metrics, candidates)
        if starved is not None:
            green = self.green_time_calculator.calculate(
                starved, current_metrics, predicted_metrics,
                isolated_approaches=isolated,
            )
            return Decision(
                phase=starved,
                green_time=self._validate_green_time(green),
                reason="starvation_override",
                starvation_override=True,
                isolated_approaches=isolated,
                candidate_approaches=candidates,
            )

        high_density = self._find_high_density_phase(current_metrics, candidates)
        if high_density is not None:
            green = self.green_time_calculator.calculate(
                high_density, current_metrics, predicted_metrics,
                isolated_approaches=isolated,
            )
            return Decision(
                phase=high_density,
                green_time=self._validate_green_time(green),
                reason="high_density_override",
                isolated_approaches=isolated,
                candidate_approaches=candidates,
            )

        phase_metrics = self._build_phase_metrics(current_metrics, predicted_metrics, candidates)
        qubo_input = {
            "junction_id": junction_id,
            "candidate_roads": list(candidates),
            "waiting_time": {p: m["waiting_time"] for p, m in phase_metrics.items()},
            "density": {p: m["density"] for p, m in phase_metrics.items()},
            "queue_storage_ratio": {p: m["queue_storage_ratio"] for p, m in phase_metrics.items()},
            "queue_length": {p: m["queue_length"] for p, m in phase_metrics.items()},
        }

        qubo = self.qubo_builder.build(qubo_input)
        total_vehicles = sum(
            int(current_metrics.get(d, {}).get("vehicle_count", 0) or 0) for d in config.DIRECTIONS
        )
        solver_result = self.solver_manager.solve(
            qubo=qubo,
            junction_id=junction_id,
            candidate_roads=list(candidates),
            solver_name=solver_name,
            vehicle_count=total_vehicles,
        )

        phase = solver_result.get("selected_phase")
        if phase not in candidates:
            raise RuntimeError(
                f"Solver returned invalid phase {phase!r}; candidates are {sorted(candidates)}."
            )

        green = self.green_time_calculator.calculate(
            phase, current_metrics, predicted_metrics,
            isolated_approaches=isolated,
        )
        return Decision(
            phase=phase,
            green_time=self._validate_green_time(green),
            reason="qubo_optimisation",
            isolated_approaches=isolated,
            candidate_approaches=candidates,
            solver_result=solver_result,
            qubo=qubo,
        )

    @staticmethod
    def _candidate_phases(isolated: Set[str]) -> Set[str]:
        return {
            phase for phase, directions in config.PHASE_MAP.items()
            if set(directions) - isolated
        }

    @staticmethod
    def _phase_available(phase: str, candidates: Set[str]) -> bool:
        return phase in candidates

    @staticmethod
    def _find_starved_phase(
        current_metrics: Dict[str, Dict[str, float]], candidates: Set[str]
    ) -> Optional[str]:
        best_phase = None
        best_wait = -1.0
        for phase in config.PHASES:
            if phase not in candidates:
                continue
            wait = max(
                float(current_metrics.get(direction, {}).get("waiting_time", 0.0) or 0.0)
                for direction in config.PHASE_MAP[phase]
            )
            if wait >= config.MAX_WAIT_SECONDS and wait > best_wait:
                best_phase, best_wait = phase, wait
        return best_phase

    @staticmethod
    def _find_high_density_phase(
        current_metrics: Dict[str, Dict[str, float]], candidates: Set[str]
    ) -> Optional[str]:
        best_phase = None
        best_density = -1.0
        for phase in config.PHASES:
            if phase not in candidates:
                continue
            density = max(
                float(current_metrics.get(direction, {}).get("density", 0.0) or 0.0)
                for direction in config.PHASE_MAP[phase]
            )
            if density >= config.DENSITY_THRESHOLD and density > best_density:
                best_phase, best_density = phase, density
        return best_phase


    @staticmethod
    def _build_phase_metrics(
        current: Dict[str, Dict[str, float]],
        predicted: Dict[str, Dict[str, float]],
        candidates: Set[str],
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate direction metrics into phase-level QUBO candidates."""
        result: Dict[str, Dict[str, float]] = {}
        for phase in config.PHASES:
            if phase not in candidates:
                continue
            directions = config.PHASE_MAP[phase]
            result[phase] = {
                "waiting_time": max(float(current.get(d, {}).get("waiting_time", 0.0) or 0.0) for d in directions),
                "queue_length": max(float(current.get(d, {}).get("queue_length", 0.0) or 0.0) for d in directions),
                "queue_storage_ratio": max(float(current.get(d, {}).get("queue_storage_ratio", 0.0) or 0.0) for d in directions),
                "density": max(float(current.get(d, {}).get("density", 0.0) or 0.0) for d in directions),
                "predicted_waiting_time": max(float(predicted.get(d, {}).get("waiting_time", 0.0) or 0.0) for d in directions),
            }
        return result

    @staticmethod
    def _validate_green_time(value: Any) -> float:
        try:
            green = float(value)
        except (TypeError, ValueError):
            green = config.MIN_GREEN_TIME
        return max(config.MIN_GREEN_TIME, min(config.MAX_GREEN_TIME, green))