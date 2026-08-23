"""Solver orchestration for the traffic-control QUBO.

This module does not build or modify a QUBO. It dispatches the exact same
QUBO to exactly ONE solver per decision cycle and returns that solver's
result.

Final project architecture: only two operational solvers exist --
Simulated Annealing (low/normal traffic) and QAOA (high/complex traffic).
The Classical exact-enumeration solver has been removed entirely from this
module, along with the old "all" multi-solver/benchmarking mode.

Modes:
    adaptive              select Simulated Annealing or QAOA based on
                           current vehicle load (see Solver Selection
                           Policy below); this is the spec-mandated runtime
                           behaviour.
    simulated_annealing   run simulated annealing directly
    qaoa                  run local QAOA directly

Explicitly forbidden and therefore NOT implemented by this module:
    - running SA + QAOA together
    - comparing solver outputs / selecting the best among several
    - keeping benchmarking logs between solvers
    - any post-failure "fallback" execution of a second solver in the same
      decision cycle (only ONE solver may execute per cycle, full stop)

Solver Selection Policy (adaptive mode)
----------------------------------------
Let N = vehicle_count (total active vehicles in the current decision
region). The threshold is read from config:

    config.HIGH_TRAFFIC_THRESHOLD

If N > HIGH_TRAFFIC_THRESHOLD:
    Use QAOA (high/complex traffic).
Otherwise:
    Use Simulated Annealing (low/normal traffic).

The selection happens once, before any solver runs. If the selected solver
fails (raises or returns success=False), that failure is returned as-is to
the caller rather than silently retried with the other solver -- fail
clearly rather than silently produce an incorrect decision.

Flow: traffic data -> determine traffic level -> select exactly one solver
-> execute it once -> return result.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from .simulated_annealing_solver import SimulatedAnnealingConfig, solve as sa_solve
from .quantum_solver import QAOASolverConfig, solve as qaoa_solve


class SolverManagerError(RuntimeError):
    """Raised when solver-manager configuration or dispatch is invalid."""


@dataclass(frozen=True)
class SolverManagerConfig:
    """Runtime configuration for single-solver dispatch."""

    solver: str = "adaptive"
    simulated_annealing: SimulatedAnnealingConfig = SimulatedAnnealingConfig()
    qaoa: QAOASolverConfig = QAOASolverConfig()

    def validate(self) -> None:
        allowed = {"simulated_annealing", "qaoa", "adaptive"}
        if self.solver not in allowed:
            raise SolverManagerError(
                f"Unsupported solver '{self.solver}'. Expected one of {sorted(allowed)}."
            )


class SolverManager:
    """Dispatches one immutable QUBO to exactly one solver per decision cycle."""

    def __init__(self, config: Optional[SolverManagerConfig] = None) -> None:
        self.config = config or SolverManagerConfig()
        self.config.validate()
        self.history = []

    def solve(
        self,
        qubo: Dict[str, Any],
        junction_id: Optional[str] = None,
        candidate_roads: Optional[Sequence[str]] = None,
        solver_name: Optional[str] = None,
        vehicle_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Solve the QUBO with exactly one solver and return its normalized result.

        Adaptive mode selects Simulated Annealing for low/normal load
        (N <= HIGH_TRAFFIC_THRESHOLD) and QAOA for high/complex load
        (N > HIGH_TRAFFIC_THRESHOLD). The selection happens before dispatch; only
        one solver is ever run.
        """
        requested = solver_name or self.config.solver
        if requested not in {"simulated_annealing", "qaoa", "adaptive"}:
            raise SolverManagerError(f"Unsupported solver: {requested}")

        if requested == "adaptive":
            requested = self._select_adaptive_solver(vehicle_count)

        result = self._run_one(requested, qubo, junction_id, candidate_roads)
        self.history.append(result)
        return result

    def _select_adaptive_solver(self, vehicle_count: Optional[int]) -> str:
        """Determines traffic level and selects exactly one solver, before
        any solver executes -- never as a post-failure retry."""
        import config as sys_config

        high_thresh = getattr(sys_config, "HIGH_TRAFFIC_THRESHOLD", 30)
        n_vehs = vehicle_count if vehicle_count is not None else 0

        if n_vehs > high_thresh:
            return "qaoa"
        return "simulated_annealing"

    def _run_one(
        self,
        solver_name: str,
        qubo: Dict[str, Any],
        junction_id: Optional[str],
        candidate_roads: Optional[Sequence[str]],
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            if solver_name == "simulated_annealing":
                result = sa_solve(
                    qubo, junction_id=junction_id, candidate_roads=candidate_roads,
                    config=self.config.simulated_annealing,
                )
            elif solver_name == "qaoa":
                result = qaoa_solve(
                    qubo, junction_id=junction_id, candidate_roads=candidate_roads,
                    config=self.config.qaoa,
                )
            else:
                raise SolverManagerError(f"Unsupported solver: {solver_name}")
        except Exception as exc:
            return {
                "solver": solver_name,
                "junction_id": junction_id,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "execution_time_ms": (time.perf_counter() - started) * 1000.0,
            }

        result = dict(result)
        result.setdefault("success", True)
        result.setdefault("execution_time_ms", (time.perf_counter() - started) * 1000.0)
        return result