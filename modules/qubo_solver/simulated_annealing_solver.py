#Solve the QUBO using Simulated Annealing.
"""
simulated_annealing_solver.py
==============

Simulated Annealing Solver for the Adaptive Alternate Green Wave
Traffic Signal Coordination System.

RESPONSIBILITY (and ONLY responsibility)
------------------------------------------
Receive the already-constructed QUBO from `qubo_builder.py` and solve
it via Simulated Annealing, returning the selected phase.

This module contains ONLY the Simulated Annealing optimization
algorithm (plus the minimal QUBO-representation parsing needed to
evaluate energy consistently with qubo_builder.py's output). It does
NOT:
    - check density or the 65% threshold
    - determine traffic mode
    - change QUBO weights
    - recalculate waiting time, density, or queue length
    - rebuild the QUBO
    - communicate with SUMO / TraCI
    - detect blockage or emergency vehicles
    - manage road isolation
    - calculate green time
    - control traffic signals

Architecture
------------
    Decision Engine -> QUBO Builder -> QUBO -> Simulated Annealing
    Solver -> Selected Phase -> Green Time Calculator -> Signal
    Controller

Current phase variables: x_NS, x_EW (exactly one must be 1). The
implementation generalizes to more binary variables (future
multi-phase junctions).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class QUBOSolverError(ValueError):
    """Raised for malformed QUBO input or invalid configuration."""


class NoValidSolutionError(RuntimeError):
    """Raised when Simulated Annealing fails to find any solution that
    satisfies the exactly-one-phase constraint."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulatedAnnealingConfig:
    """
    Tunable Simulated Annealing parameters. Nothing about the
    annealing schedule or search behavior is hard-coded outside of
    this config.

    initial_temperature: starting temperature T0 (> 0).
    final_temperature: floor temperature; annealing for a given
        restart stops early once T drops below this value (> 0,
        and < initial_temperature).
    cooling_rate: geometric cooling factor applied each iteration,
        T_(k+1) = T_k * cooling_rate (must be in (0, 1)).
    iterations: maximum number of SA iterations per restart (> 0).
    restarts: number of independent annealing runs; the best valid
        solution across all restarts is returned (>= 1).
    random_seed: seed for reproducibility. When set, identical seed +
        configuration + QUBO reproduce identical results, since a
        dedicated `random.Random` instance (not the global RNG) is
        used throughout.
    """
    initial_temperature: float = 10.0
    final_temperature: float = 0.01
    cooling_rate: float = 0.95
    iterations: int = 1000
    restarts: int = 5
    random_seed: Optional[int] = None

    def validate(self) -> None:
        if self.initial_temperature <= 0:
            raise QUBOSolverError(
                f"initial_temperature must be positive, got {self.initial_temperature}."
            )
        if self.final_temperature <= 0:
            raise QUBOSolverError(
                f"final_temperature must be positive, got {self.final_temperature}."
            )
        if self.final_temperature >= self.initial_temperature:
            raise QUBOSolverError(
                "final_temperature must be strictly less than initial_temperature "
                f"(got final={self.final_temperature}, initial={self.initial_temperature})."
            )
        if not (0.0 < self.cooling_rate < 1.0):
            raise QUBOSolverError(
                f"cooling_rate must be in (0, 1), got {self.cooling_rate}."
            )
        if self.iterations <= 0:
            raise QUBOSolverError(f"iterations must be positive, got {self.iterations}.")
        if self.restarts <= 0:
            raise QUBOSolverError(f"restarts must be positive, got {self.restarts}.")


# ---------------------------------------------------------------------------
# Internal canonical QUBO representation
# (Mirrors the representation used by qubo_builder.py's output;
#  duplicated here rather than imported so this module's only
#  external dependency is the QUBO dict itself.)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CanonicalQUBO:
    variables: List[str]
    linear: Dict[str, float]
    quadratic: Dict[Tuple[str, str], float]   # each unordered pair appears once
    constant: float


def _split_quadratic_string_key(key: str, known_variables: Sequence[str]) -> Tuple[str, str]:
    """Resolve a serialized 'var_i_var_j' key against known variable names,
    to correctly handle variable names that themselves contain underscores."""
    for var_a in known_variables:
        for var_b in known_variables:
            if var_a == var_b:
                continue
            if key == f"{var_a}_{var_b}":
                return var_a, var_b
    raise QUBOSolverError(
        f"Could not resolve quadratic term key '{key}' against known variables {list(known_variables)}."
    )


def _infer_variables(qubo: Dict, candidate_roads: Optional[Sequence[str]]) -> List[str]:
    """
    Determine ordered decision-variable names, preferring (in order):
      1. qubo["candidate_phases"]  (qubo_builder.py's canonical field)
      2. qubo["linear_terms"] keys
      3. candidate_roads argument  (legacy/back-compat interface)
      4. qubo["qubo_matrix"] dimension, with generic names
    """
    if qubo.get("candidate_phases"):
        variables = list(qubo["candidate_phases"])
    elif qubo.get("linear_terms"):
        variables = list(qubo["linear_terms"].keys())
    elif candidate_roads:
        variables = list(candidate_roads)
    elif qubo.get("qubo_matrix"):
        n = len(qubo["qubo_matrix"])
        variables = [f"x{i}" for i in range(n)]
    else:
        raise QUBOSolverError(
            "Unable to determine decision variables from the supplied QUBO "
            "(no candidate_phases, linear_terms, candidate_roads, or qubo_matrix found)."
        )

    if len(variables) < 2:
        raise QUBOSolverError(
            f"At least 2 candidate phases are required for the exactly-one constraint, "
            f"got {len(variables)}: {variables}."
        )
    if len(set(variables)) != len(variables):
        raise QUBOSolverError(f"Duplicate candidate phase names found: {variables}.")

    return variables


def _canonicalize_from_linear_quadratic(qubo: Dict, variables: List[str]) -> _CanonicalQUBO:
    linear_raw = qubo.get("linear_terms", {}) or {}
    linear = {var: float(linear_raw.get(var, 0.0)) for var in variables}

    quadratic_raw = qubo.get("quadratic_terms", {}) or {}
    quadratic: Dict[Tuple[str, str], float] = {}
    for key, value in quadratic_raw.items():
        if isinstance(key, tuple):
            if len(key) != 2:
                raise QUBOSolverError(f"Quadratic term key must have exactly 2 variables, got: {key}")
            var_i, var_j = key
        elif isinstance(key, str):
            var_i, var_j = _split_quadratic_string_key(key, variables)
        else:
            raise QUBOSolverError(f"Unsupported quadratic term key type: {type(key)}")

        if var_i not in variables or var_j not in variables:
            raise QUBOSolverError(f"Quadratic term references unknown variable(s): {key}")
        if var_i == var_j:
            raise QUBOSolverError(
                f"Quadratic term ({var_i}, {var_j}) references the same variable twice; "
                "diagonal terms belong in linear_terms, not quadratic_terms."
            )
        pair = tuple(sorted((var_i, var_j)))
        if pair in quadratic:
            raise QUBOSolverError(f"Duplicate quadratic term for pair {pair}; each pair must appear once.")
        quadratic[pair] = float(value)

    constant = float(qubo.get("constant_offset", 0.0))
    return _CanonicalQUBO(variables=variables, linear=linear, quadratic=quadratic, constant=constant)


def _canonicalize_from_matrix(qubo: Dict, variables: List[str]) -> _CanonicalQUBO:
    """
    Fallback when linear_terms / quadratic_terms are absent. Handles
    both fully-off-diagonal and symmetrically-split matrix
    conventions by summing M[i][j] + M[j][i] per pair, which is
    correct either way and avoids double counting.
    """
    matrix = qubo["qubo_matrix"]
    n = len(variables)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise QUBOSolverError(
            f"qubo_matrix shape {len(matrix)}x{len(matrix[0]) if matrix else 0} "
            f"does not match variable count {n}."
        )

    linear = {variables[i]: float(matrix[i][i]) for i in range(n)}
    quadratic: Dict[Tuple[str, str], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            combined = float(matrix[i][j]) + float(matrix[j][i])
            if combined != 0.0:
                quadratic[(variables[i], variables[j])] = combined

    constant = float(qubo.get("constant_offset", 0.0))
    return _CanonicalQUBO(variables=variables, linear=linear, quadratic=quadratic, constant=constant)


def _canonicalize_qubo(qubo: Dict, candidate_roads: Optional[Sequence[str]]) -> _CanonicalQUBO:
    if qubo is None:
        raise QUBOSolverError("No QUBO was supplied (qubo is None).")
    if not isinstance(qubo, dict) or not qubo:
        raise QUBOSolverError("Supplied QUBO is empty or not a dict.")

    variables = _infer_variables(qubo, candidate_roads)

    has_linear_quadratic = bool(qubo.get("linear_terms")) or bool(qubo.get("quadratic_terms"))
    if has_linear_quadratic:
        return _canonicalize_from_linear_quadratic(qubo, variables)

    if qubo.get("qubo_matrix"):
        return _canonicalize_from_matrix(qubo, variables)

    raise QUBOSolverError(
        "Supplied QUBO has neither linear_terms/quadratic_terms nor a qubo_matrix to solve."
    )


def _evaluate_energy(canonical: _CanonicalQUBO, assignment: Dict[str, int]) -> float:
    """
    E(x) = sum_i linear_i * x_i + sum_{i<j} quadratic_(i,j) * x_i * x_j + constant.

    Each unordered pair in `canonical.quadratic` is stored exactly
    once by construction, so no factor-of-2 correction is needed here.
    """
    energy = canonical.constant
    for var, coeff in canonical.linear.items():
        energy += coeff * assignment[var]
    for (var_i, var_j), coeff in canonical.quadratic.items():
        energy += coeff * assignment[var_i] * assignment[var_j]
    return energy


def _is_exactly_one_hot(assignment: Dict[str, int]) -> bool:
    return sum(assignment.values()) == 1


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------

def _random_one_hot_assignment(variables: List[str], rng: random.Random) -> Dict[str, int]:
    """A random valid state: exactly one variable set to 1."""
    chosen = rng.choice(variables)
    return {var: (1 if var == chosen else 0) for var in variables}


def _flip_neighbor(assignment: Dict[str, int], variables: List[str], rng: random.Random) -> Dict[str, int]:
    """
    Generate a neighboring binary solution by flipping a single
    randomly-chosen bit. This is the standard SA move for binary
    QUBOs and generalizes to any number of variables; it can produce
    constraint-violating states (e.g. [0,0] or [1,1] for n=2), which
    is expected — the QUBO's penalty term makes such states
    energetically unfavorable, and this solver separately tracks only
    the best *valid* state encountered (see `_anneal_single_restart`).
    Note that a single-bit flip never moves directly between two valid
    one-hot states (e.g. NS -> EW always passes through an invalid
    corner such as [0,0] or [1,1] first); reaching the other valid
    state relies on the penalty being low enough to occasionally accept
    that intermediate step, plus the restarts in `_anneal`, which sample
    every one-hot state directly as a starting point.
    """
    neighbor = dict(assignment)
    flip_var = rng.choice(variables)
    neighbor[flip_var] = 1 - neighbor[flip_var]
    return neighbor


def _anneal_single_restart(
    canonical: _CanonicalQUBO,
    config: SimulatedAnnealingConfig,
    rng: random.Random,
) -> Tuple[Optional[Dict[str, int]], Optional[float], float, int]:
    """
    Run one Simulated Annealing restart.

    Returns:
        (best_valid_assignment, best_valid_energy, final_temperature_reached,
         iterations_run)

    best_valid_assignment/energy may be None only in pathological
    cases (never in normal operation, since the initial state is
    itself always a valid one-hot assignment).

    iterations_run is the actual number of SA iterations executed before
    either the iteration cap or the final_temperature floor was reached --
    it can be well below config.iterations when cooling reaches the floor
    early, and callers must not assume it equals config.iterations.
    """
    current = _random_one_hot_assignment(canonical.variables, rng)
    current_energy = _evaluate_energy(canonical, current)

    best_valid_assignment: Optional[Dict[str, int]] = None
    best_valid_energy: Optional[float] = None
    if _is_exactly_one_hot(current):
        best_valid_assignment = dict(current)
        best_valid_energy = current_energy

    temperature = config.initial_temperature
    iterations_run = 0

    for _ in range(config.iterations):
        if temperature < config.final_temperature:
            break
        iterations_run += 1

        neighbor = _flip_neighbor(current, canonical.variables, rng)
        neighbor_energy = _evaluate_energy(canonical, neighbor)
        delta = neighbor_energy - current_energy

        if delta <= 0:
            accept = True
        else:
            probability = math.exp(-delta / temperature)
            accept = rng.random() < probability

        if accept:
            current, current_energy = neighbor, neighbor_energy

            if _is_exactly_one_hot(current):
                if best_valid_energy is None or current_energy < best_valid_energy:
                    best_valid_assignment = dict(current)
                    best_valid_energy = current_energy

        temperature *= config.cooling_rate

    return best_valid_assignment, best_valid_energy, temperature, iterations_run


def _anneal(
    canonical: _CanonicalQUBO,
    config: SimulatedAnnealingConfig,
) -> Tuple[Dict[str, int], float, float, int]:
    """
    Run all configured restarts and return the best valid solution
    found across all of them.

    Returns:
        (best_assignment, best_energy, final_temperature_of_best_restart,
         iterations_run_of_best_restart)
    """
    rng = random.Random(config.random_seed)

    global_best_assignment: Optional[Dict[str, int]] = None
    global_best_energy: Optional[float] = None
    global_final_temperature: float = config.initial_temperature
    global_iterations_run: int = 0

    for _ in range(config.restarts):
        assignment, energy, final_temp, iterations_run = _anneal_single_restart(
            canonical, config, rng
        )
        if assignment is None or energy is None:
            continue
        if global_best_energy is None or energy < global_best_energy:
            global_best_assignment = assignment
            global_best_energy = energy
            global_final_temperature = final_temp
            global_iterations_run = iterations_run

    if global_best_assignment is None or global_best_energy is None:
        raise NoValidSolutionError(
            "Simulated Annealing did not find any solution satisfying the "
            "exactly-one-phase constraint across all restarts."
        )

    return global_best_assignment, global_best_energy, global_final_temperature, global_iterations_run


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def solve(
    qubo: Dict,
    junction_id: Optional[str] = None,
    candidate_roads: Optional[Sequence[str]] = None,
    config: Optional[SimulatedAnnealingConfig] = None,
) -> Dict:
    """
    Solve a QUBO produced by qubo_builder.py using Simulated
    Annealing and return the selected phase.

    This function does NOT know or care which traffic-priority mode
    (NORMAL / HIGH_DENSITY) was used to construct the QUBO — it only
    minimizes the energy function it is given and enforces the
    exactly-one-phase constraint on its final answer.

    Args:
        qubo: The QUBO dict from qubo_builder.py (or an equivalent
            structure exposing candidate_phases/linear_terms/
            quadratic_terms/constant_offset, or a qubo_matrix).
        junction_id: Optional junction identifier; if omitted, taken
            from qubo["junction_id"] when present.
        candidate_roads: Optional legacy/back-compat interface
            parameter, consulted only if the QUBO itself does not
            already specify candidate_phases/linear_terms.
        config: Simulated Annealing configuration; defaults to
            SimulatedAnnealingConfig() if omitted.

    Returns:
        {
            "solver": "simulated_annealing",
            "junction_id": ...,
            "selected_phase": "NS" or "EW",
            "binary_solution": [1, 0] or [0, 1],
            "best_energy": ...,
            "constraint_valid": True,
            "iterations": ...,       # actual iterations run by the winning
                                      # restart before hitting the iteration
                                      # cap or the final_temperature floor
                                      # (may be well below config.iterations)
            "restarts": ...,         # configured restart count
            "initial_temperature": ...,
            "final_temperature": ...,  # actual temperature reached by the
                                        # winning restart, NOT the configured
                                        # floor -- these can differ
            "execution_time_ms": ...,
            "random_seed": ...
        }

    Raises:
        QUBOSolverError: for missing/empty/malformed QUBO input or
            invalid configuration.
        NoValidSolutionError: if annealing fails to find any solution
            satisfying the exactly-one-phase constraint.
    """
    config = config or SimulatedAnnealingConfig()
    config.validate()

    resolved_junction_id = junction_id if junction_id is not None else (qubo or {}).get("junction_id")

    start = time.perf_counter()

    canonical = _canonicalize_qubo(qubo, candidate_roads)
    best_assignment, best_energy, final_temperature_reached, iterations_run = _anneal(
        canonical, config
    )

    constraint_valid = _is_exactly_one_hot(best_assignment)
    if not constraint_valid:
        # Should be unreachable given _anneal only returns one-hot
        # assignments, but validated explicitly per the spec's
        # requirement that the solver never silently returns an
        # invalid phase.
        raise NoValidSolutionError(
            f"Best solution found violates the exactly-one-phase constraint: {best_assignment}"
        )

    selected_phase = next(var for var, value in best_assignment.items() if value == 1)
    binary_solution = [best_assignment[var] for var in canonical.variables]

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return {
        "solver": "simulated_annealing",
        "junction_id": resolved_junction_id,
        "selected_phase": selected_phase,
        "binary_solution": binary_solution,
        "best_energy": best_energy,
        "constraint_valid": constraint_valid,
        "iterations": iterations_run,
        "restarts": config.restarts,
        "initial_temperature": config.initial_temperature,
        "final_temperature": final_temperature_reached,
        "execution_time_ms": elapsed_ms,
        "random_seed": config.random_seed,
    }


# ---------------------------------------------------------------------------
# Example usage (manual smoke test; not executed on import)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # NS is more congested than EW -> NS should win.
    example_qubo = {
        "junction_id": "J1",
        "candidate_phases": ["NS", "EW"],
        "linear_terms": {"NS": -11.0, "EW": -10.0},
        "quadratic_terms": {"NS_EW": 20.0},
        "constant_offset": 10.0,
        "qubo_matrix": [[-11.0, 10.0], [10.0, -10.0]],
    }

    result = solve(example_qubo, config=SimulatedAnnealingConfig(random_seed=42))
    print(json.dumps(result, indent=2))
    assert result["selected_phase"] == "NS"
    assert result["binary_solution"] == [1, 0]
    assert result["constraint_valid"] is True

    # Reproducibility check: same seed + config + QUBO -> same result.
    result_repeat = solve(example_qubo, config=SimulatedAnnealingConfig(random_seed=42))
    assert result_repeat["selected_phase"] == result["selected_phase"]
    assert result_repeat["binary_solution"] == result["binary_solution"]
    assert result_repeat["best_energy"] == result["best_energy"]

    # EW is more congested -> EW should win.
    example_qubo_2 = {
        "linear_terms": {"NS": -5.0, "EW": -8.0},
        "quadratic_terms": {("NS", "EW"): 20.0},
        "constant_offset": 10.0,
    }
    result2 = solve(example_qubo_2, junction_id="J2", candidate_roads=["NS", "EW"],
                     config=SimulatedAnnealingConfig(random_seed=7))
    print(json.dumps(result2, indent=2))
    assert result2["selected_phase"] == "EW"
    assert result2["binary_solution"] == [0, 1]

    # Error handling checks.
    try:
        solve(None)
        raise AssertionError("Expected QUBOSolverError for missing QUBO")
    except QUBOSolverError:
        pass

    try:
        solve({})
        raise AssertionError("Expected QUBOSolverError for empty QUBO")
    except QUBOSolverError:
        pass

    try:
        solve(example_qubo, config=SimulatedAnnealingConfig(cooling_rate=1.5))
        raise AssertionError("Expected QUBOSolverError for invalid cooling_rate")
    except QUBOSolverError:
        pass

    print("All smoke tests passed.")