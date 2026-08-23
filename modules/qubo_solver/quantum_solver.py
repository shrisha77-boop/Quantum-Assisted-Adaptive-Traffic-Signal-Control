#Solve the QUBO using a quantum algorithm.(QAOA)
"""
qaoa_solver.py
==============

QAOA (Quantum Approximate Optimization Algorithm) Solver for the
Adaptive Alternate Green Wave Traffic Signal Coordination System.

RESPONSIBILITY (and ONLY responsibility)
------------------------------------------
Receive the already-constructed QUBO from `qubo_builder.py`, convert
it to an equivalent Ising cost Hamiltonian, run QAOA on a local
Qiskit Aer simulator to find candidate solutions, validate those
solutions against the exactly-one-phase constraint, and return the
lowest-energy valid solution (evaluated against the ORIGINAL QUBO).

This module does NOT:
    - check density or the 65% threshold
    - determine traffic mode or priority order
    - change QUBO weights or rebuild the traffic objective
    - recalculate waiting time, density, or queue length
    - read TraCI, collect traffic data, detect blockage/emergency
      vehicles, or manage road isolation
    - calculate green time
    - control traffic signals (never calls any traci.trafficlight.* API)

Architecture (QAOA branch only)
--------------------------------
    Decision Engine -> qubo_builder.py -> SAME QUBO -> qaoa_solver.py
    -> QUBO->Ising -> QAOA circuit -> Qiskit Aer (local simulator)
    -> measurement -> valid bitstrings -> original QUBO energy
    evaluation -> best valid solution -> NS or EW -> Green Time
    Calculator -> Signal Controller

Dependencies
------------
Qiskit and Qiskit Aer are imported LAZILY, inside `solve()`, so that
importing this module (or running the Simulated Annealing solver)
never requires Qiskit to be installed. Qiskit is only required when
`solve()` is actually invoked.

Local simulation only
----------------------
The only supported backend is a local Qiskit Aer simulator
("local_aer_simulator"). No IBM Quantum account, API key, cloud
authentication, internet connection, or paid quantum hardware is
required or used. `QAOASolverConfig.backend` is an explicit
extension point for a future hardware/cloud backend, but any value
other than "local_aer_simulator" currently raises a clear error
rather than silently falling back.

Bitstring / qubit convention
------------------------------
For decision variables ordered as `variables = [v0, v1, ..., v_{n-1}]`
(taken from the QUBO's own variable ordering — see `_infer_variables`):

    qubit index q  <->  variables[q]

Qiskit reports measurement bitstrings with qubit 0 as the RIGHTMOST
character (little-endian). So for a bitstring string `s` of length n:

    bit for variables[q]  =  int(s[n - 1 - q])

This convention is applied consistently in Pauli-string construction
(`_pauli_label`), circuit construction (QAOAAnsatz over the same
qubit indices), and measurement extraction
(`_bitstring_to_assignment`). For the current 2-variable problem this
means: bit 0 (rightmost character) -> variables[0] (typically "NS"),
bit 1 -> variables[1] (typically "EW"), matching the ordering
`qubo["candidate_phases"]` supplies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class QAOASolverError(ValueError):
    """Raised for malformed QUBO input, invalid configuration, or
    missing/broken quantum dependencies."""


class NoValidQAOASolutionError(RuntimeError):
    """Raised when no measured bitstring satisfies the exactly-one-phase
    constraint and no fallback is configured."""


class QAOAOptimizationError(RuntimeError):
    """Raised when the classical parameter-optimization loop fails."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SUPPORTED_OPTIMIZERS = frozenset({"COBYLA", "Nelder-Mead", "Powell", "L-BFGS-B", "SLSQP"})
_SUPPORTED_BACKENDS = frozenset({"local_aer_simulator"})


@dataclass(frozen=True)
class QAOASolverConfig:
    """
    Tunable QAOA parameters. Nothing about circuit depth, shot count,
    or the optimizer is hard-coded outside of this config.

    qaoa_depth (p): number of QAOA layers (>= 1).
    shots: number of circuit executions for the final measurement (>= 1).
    optimizer: classical optimizer name, passed to scipy.optimize.minimize.
        One of: COBYLA, Nelder-Mead, Powell, L-BFGS-B, SLSQP.
    max_optimizer_iterations: iteration cap for the classical optimizer (>= 1).
    initial_parameters: optional explicit starting point for [beta..., gamma...]
        (length must equal 2 * qaoa_depth). If omitted, a seeded random
        starting point in [0, 2*pi) is used.
    random_seed: seed for the initial-parameter draw and the Aer simulator.
        Same QUBO + config + seed => reproducible results, as far as the
        underlying Qiskit/Aer/scipy algorithms permit.
    backend: execution backend identifier. Only "local_aer_simulator" is
        currently supported; this field exists as an extension point for a
        future backend without requiring changes to QUBO construction or
        this solver's public interface.
    fallback_on_no_valid_solution: if True and NO measured bitstring
        satisfies the exactly-one-phase constraint, fall back to
        evaluating the original QUBO energy directly over the valid
        one-hot assignments (bypassing the quantum measurement) rather
        than raising. This is a clearly documented, explicit fallback,
        never a silent one — the returned result marks
        `used_fallback: True` when triggered. Defaults to False (raise).
    """
    qaoa_depth: int = 1
    shots: int = 2000
    optimizer: str = "COBYLA"
    max_optimizer_iterations: int = 200
    initial_parameters: Optional[List[float]] = None
    random_seed: Optional[int] = None
    backend: str = "local_aer_simulator"
    fallback_on_no_valid_solution: bool = False

    def validate(self) -> None:
        if not isinstance(self.qaoa_depth, int) or self.qaoa_depth < 1:
            raise QAOASolverError(f"qaoa_depth must be a positive integer, got {self.qaoa_depth!r}.")
        if not isinstance(self.shots, int) or self.shots < 1:
            raise QAOASolverError(f"shots must be a positive integer, got {self.shots!r}.")
        if self.optimizer not in _SUPPORTED_OPTIMIZERS:
            raise QAOASolverError(
                f"Unsupported optimizer '{self.optimizer}'. Supported: {sorted(_SUPPORTED_OPTIMIZERS)}."
            )
        if not isinstance(self.max_optimizer_iterations, int) or self.max_optimizer_iterations < 1:
            raise QAOASolverError(
                f"max_optimizer_iterations must be a positive integer, got {self.max_optimizer_iterations!r}."
            )
        if self.initial_parameters is not None:
            expected_len = 2 * self.qaoa_depth
            if len(self.initial_parameters) != expected_len:
                raise QAOASolverError(
                    f"initial_parameters must have length 2 * qaoa_depth = {expected_len}, "
                    f"got {len(self.initial_parameters)}."
                )
        if self.backend not in _SUPPORTED_BACKENDS:
            raise QAOASolverError(
                f"Unsupported backend '{self.backend}'. Currently supported: "
                f"{sorted(_SUPPORTED_BACKENDS)}. (Extension point for future backends; "
                "the local Aer simulator remains the mandatory default.)"
            )


# ---------------------------------------------------------------------------
# Internal canonical QUBO representation
# (Mirrors qubo_solver.py, the simulated annealing solver, so both
#  operational solvers -- QAOA and Simulated Annealing -- interpret an
#  identical QUBO identically.)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CanonicalQUBO:
    variables: List[str]
    linear: Dict[str, float]
    quadratic: Dict[Tuple[str, str], float]   # each unordered pair appears once
    constant: float


def _split_quadratic_string_key(key: str, known_variables: Sequence[str]) -> Tuple[str, str]:
    for var_a in known_variables:
        for var_b in known_variables:
            if var_a == var_b:
                continue
            if key == f"{var_a}_{var_b}":
                return var_a, var_b
    raise QAOASolverError(
        f"Could not resolve quadratic term key '{key}' against known variables {list(known_variables)}."
    )


def _infer_variables(qubo: Dict, candidate_roads: Optional[Sequence[str]]) -> List[str]:
    """
    Determine ordered decision-variable names, preferring (in order):
      1. qubo["candidate_phases"]  (qubo_builder.py's canonical field)
      2. qubo["linear_terms"] keys
      3. candidate_roads argument  (legacy/back-compat interface)
      4. qubo["qubo_matrix"] dimension, with generic names

    This ordering also fixes the qubit index <-> variable mapping used
    throughout Ising conversion, circuit construction, and measurement
    extraction (see module docstring).

    If candidate_roads is ALSO supplied (e.g. the caller's own
    understanding of the current candidate set, passed alongside the
    QUBO) and the QUBO's own embedded variables were used instead (paths
    1 or 2 above), the two must agree. A mismatch means a stale or
    wrong-junction QUBO was handed to this solver -- exactly the
    "Mismatched candidate roads" condition that must raise a clear error
    rather than be silently ignored.
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
        raise QAOASolverError(
            "Unable to determine decision variables from the supplied QUBO "
            "(no candidate_phases, linear_terms, candidate_roads, or qubo_matrix found)."
        )

    if len(variables) < 2:
        raise QAOASolverError(
            f"At least 2 candidate phases are required for the exactly-one constraint, "
            f"got {len(variables)}: {variables}."
        )
    if len(set(variables)) != len(variables):
        raise QAOASolverError(f"Duplicate candidate phase names found: {variables}.")

    if candidate_roads and set(variables) != set(candidate_roads):
        raise QAOASolverError(
            "Mismatched candidate roads: the QUBO's own variables "
            f"{sorted(variables)} do not match the candidate_roads argument "
            f"{sorted(set(candidate_roads))}. This usually means a stale or "
            "wrong-junction QUBO was passed to the solver -- refusing to "
            "silently solve against an unverified candidate set."
        )

    return variables


def _canonicalize_from_linear_quadratic(qubo: Dict, variables: List[str]) -> _CanonicalQUBO:
    linear_raw = qubo.get("linear_terms", {}) or {}
    linear = {var: float(linear_raw.get(var, 0.0)) for var in variables}

    quadratic_raw = qubo.get("quadratic_terms", {}) or {}
    quadratic: Dict[Tuple[str, str], float] = {}
    for key, value in quadratic_raw.items():
        if isinstance(key, tuple):
            if len(key) != 2:
                raise QAOASolverError(f"Quadratic term key must have exactly 2 variables, got: {key}")
            var_i, var_j = key
        elif isinstance(key, str):
            var_i, var_j = _split_quadratic_string_key(key, variables)
        else:
            raise QAOASolverError(f"Unsupported quadratic term key type: {type(key)}")

        if var_i not in variables or var_j not in variables:
            raise QAOASolverError(f"Quadratic term references unknown variable(s): {key}")
        if var_i == var_j:
            raise QAOASolverError(
                f"Quadratic term ({var_i}, {var_j}) references the same variable twice; "
                "diagonal terms belong in linear_terms, not quadratic_terms."
            )
        pair = tuple(sorted((var_i, var_j)))
        if pair in quadratic:
            raise QAOASolverError(f"Duplicate quadratic term for pair {pair}; each pair must appear once.")
        quadratic[pair] = float(value)

    constant = float(qubo.get("constant_offset", 0.0))
    return _CanonicalQUBO(variables=variables, linear=linear, quadratic=quadratic, constant=constant)


def _canonicalize_from_matrix(qubo: Dict, variables: List[str]) -> _CanonicalQUBO:
    matrix = qubo["qubo_matrix"]
    n = len(variables)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise QAOASolverError(
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
        raise QAOASolverError("No QUBO was supplied (qubo is None).")
    if not isinstance(qubo, dict) or not qubo:
        raise QAOASolverError("Supplied QUBO is empty or not a dict.")

    variables = _infer_variables(qubo, candidate_roads)

    has_linear_quadratic = bool(qubo.get("linear_terms")) or bool(qubo.get("quadratic_terms"))
    if has_linear_quadratic:
        return _canonicalize_from_linear_quadratic(qubo, variables)

    if qubo.get("qubo_matrix"):
        return _canonicalize_from_matrix(qubo, variables)

    raise QAOASolverError(
        "Supplied QUBO has neither linear_terms/quadratic_terms nor a qubo_matrix to solve."
    )


def _evaluate_qubo_energy(canonical: _CanonicalQUBO, assignment: Dict[str, int]) -> float:
    """
    E(x) = sum_i linear_i * x_i + sum_{i<j} quadratic_(i,j) * x_i * x_j + constant.

    This is the authoritative energy function used to rank ALL
    candidate solutions (measured or fallback) — QAOA's own cost
    Hamiltonian expectation is only used internally to drive the
    classical parameter optimization loop; final selection always
    goes back through this function.
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
# QUBO -> Ising conversion
# ---------------------------------------------------------------------------

def _qubo_to_ising(canonical: _CanonicalQUBO) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], float]:
    """
    Convert the binary QUBO into an equivalent Ising Hamiltonian using
    the standard substitution x_i = (1 - z_i) / 2, i.e. x_i=0 -> z_i=+1,
    x_i=1 -> z_i=-1.

    Derivation (kept here for auditability):

        E(x) = sum_i L_i x_i + sum_{i<j} Q_ij x_i x_j + C

        Substituting x_i = (1 - z_i)/2:

        L_i x_i        = L_i/2 - (L_i/2) z_i
        Q_ij x_i x_j   = Q_ij/4 - (Q_ij/4) z_i - (Q_ij/4) z_j + (Q_ij/4) z_i z_j

    Collecting terms gives H(z) = C_ising + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j with:

        h_i      = -L_i/2 - (1/4) * sum_{j : (i,j) in quadratic} Q_ij
        J_ij     = Q_ij / 4
        C_ising  = C + sum_i L_i/2 + sum_{i<j} Q_ij/4

    This is an EXACT algebraic substitution (not an approximation), so
    H(z(x)) == E(x) for every x, which guarantees the Ising energy
    ranks every candidate solution identically to the original QUBO
    (verified numerically in this module's test suite).

    Returns:
        (h, J, ising_constant)
    """
    h: Dict[str, float] = {var: 0.0 for var in canonical.variables}
    j_coeffs: Dict[Tuple[str, str], float] = {}
    ising_constant = canonical.constant

    for var, coeff in canonical.linear.items():
        ising_constant += coeff / 2.0
        h[var] += -coeff / 2.0

    for (var_i, var_j), coeff in canonical.quadratic.items():
        ising_constant += coeff / 4.0
        h[var_i] += -coeff / 4.0
        h[var_j] += -coeff / 4.0
        j_coeffs[(var_i, var_j)] = coeff / 4.0

    return h, j_coeffs, ising_constant


def _pauli_label(num_qubits: int, active: Dict[int, str]) -> str:
    """
    Build a Qiskit Pauli-string label of length `num_qubits` with the
    given single-character Pauli operators placed at the given qubit
    indices, honoring Qiskit's convention that qubit 0 is the
    RIGHTMOST character in the label.
    """
    chars = ["I"] * num_qubits
    for qubit_index, pauli_char in active.items():
        chars[num_qubits - 1 - qubit_index] = pauli_char
    return "".join(chars)


def _build_cost_operator(
    variables: List[str],
    h: Dict[str, float],
    j_coeffs: Dict[Tuple[str, str], float],
    sparse_pauli_op_cls,
):
    """
    Build the SparsePauliOp cost Hamiltonian (WITHOUT the constant
    offset — SparsePauliOp represents an operator, not an affine
    function; the constant is added back separately wherever the
    expectation value is used).
    """
    var_index = {var: i for i, var in enumerate(variables)}
    num_qubits = len(variables)
    pauli_list: List[Tuple[str, float]] = []

    for var, coeff in h.items():
        if coeff != 0.0:
            label = _pauli_label(num_qubits, {var_index[var]: "Z"})
            pauli_list.append((label, coeff))

    for (var_i, var_j), coeff in j_coeffs.items():
        if coeff != 0.0:
            label = _pauli_label(num_qubits, {var_index[var_i]: "Z", var_index[var_j]: "Z"})
            pauli_list.append((label, coeff))

    if not pauli_list:
        # Degenerate QUBO with an all-zero linear/quadratic part; a
        # zero operator still needs a valid Pauli term to construct.
        pauli_list.append(("I" * num_qubits, 0.0))

    return sparse_pauli_op_cls.from_list(pauli_list)


def _bitstring_to_assignment(bitstring: str, variables: List[str]) -> Dict[str, int]:
    """
    Convert a Qiskit measurement bitstring into a {variable: 0/1}
    assignment, per the module's fixed convention: qubit index q
    (=> variables[q]) is read from position (len(bitstring) - 1 - q),
    since Qiskit reports qubit 0 as the rightmost character.
    """
    num_qubits = len(variables)
    if len(bitstring) != num_qubits:
        raise QAOASolverError(
            f"Invalid bitstring '{bitstring}': expected length {num_qubits}, got {len(bitstring)}."
        )
    assignment = {}
    for q, var in enumerate(variables):
        assignment[var] = int(bitstring[num_qubits - 1 - q])
    return assignment


# ---------------------------------------------------------------------------
# QAOA circuit, optimization, and measurement (Qiskit-dependent)
# ---------------------------------------------------------------------------

def _import_qiskit_dependencies():
    """
    Lazily import Qiskit / Qiskit Aer / SciPy, so that this module can
    be imported — and the Simulated Annealing solver can run — without
    Qiskit installed. Raises a clear QAOASolverError if any required
    dependency is missing.
    """
    try:
        from qiskit.circuit.library import QAOAAnsatz
        from qiskit.quantum_info import SparsePauliOp
        from qiskit import transpile
    except ImportError as exc:
        raise QAOASolverError(
            "Qiskit is required to run the QAOA solver but is not installed. "
            "Install it with: pip install qiskit"
        ) from exc

    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import EstimatorV2, SamplerV2
    except ImportError as exc:
        raise QAOASolverError(
            "Qiskit Aer is required to run the QAOA solver but is not installed. "
            "Install it with: pip install qiskit-aer"
        ) from exc

    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise QAOASolverError(
            "SciPy is required for QAOA classical parameter optimization but is not "
            "installed. Install it with: pip install scipy"
        ) from exc

    return {
        "QAOAAnsatz": QAOAAnsatz,
        "SparsePauliOp": SparsePauliOp,
        "transpile": transpile,
        "AerSimulator": AerSimulator,
        "EstimatorV2": EstimatorV2,
        "SamplerV2": SamplerV2,
        "minimize": minimize,
    }


def _initial_parameters(config: QAOASolverConfig, num_parameters: int, rng) -> List[float]:
    if config.initial_parameters is not None:
        return list(config.initial_parameters)
    return list(rng.uniform(0.0, 2.0 * math.pi, size=num_parameters))


def _optimize_qaoa_parameters(
    deps: dict,
    ansatz,
    cost_operator,
    ising_constant: float,
    config: QAOASolverConfig,
    rng,
) -> Tuple[List[float], float, int]:
    """
    Classically optimize the QAOA (beta, gamma) parameters to minimize
    the expectation value of the ORIGINAL QUBO objective (the exact
    cost Hamiltonian expectation, plus the constant offset, computed
    via the Aer Estimator primitive — no measurement sampling noise in
    this loop).

    Returns:
        (optimal_parameters, optimal_objective_value, iterations_used)

    Raises:
        QAOAOptimizationError: if the optimizer fails to complete.
    """
    estimator = deps["EstimatorV2"]()

    def objective(params: List[float]) -> float:
        job = estimator.run([(ansatz, cost_operator, params)])
        result = job.result()
        expectation_value = float(result[0].data.evs)
        return expectation_value + ising_constant

    x0 = _initial_parameters(config, ansatz.num_parameters, rng)

    try:
        result = deps["minimize"](
            objective,
            x0=x0,
            method=config.optimizer,
            options={"maxiter": config.max_optimizer_iterations},
        )
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain-specific error
        raise QAOAOptimizationError(f"QAOA classical parameter optimization failed: {exc}") from exc

    if result is None or not hasattr(result, "x") or result.x is None:
        raise QAOAOptimizationError("QAOA classical optimizer returned no usable result.")

    iterations_used = int(getattr(result, "nit", getattr(result, "nfev", 0)) or 0)
    return list(result.x), float(result.fun), iterations_used


def _sample_measurement_counts(
    deps: dict,
    ansatz,
    optimal_parameters: List[float],
    config: QAOASolverConfig,
    backend,
) -> Dict[str, int]:
    """
    Execute the optimized QAOA circuit on the local Aer simulator and
    return raw measurement counts, keyed by bitstring.
    """
    measured_circuit = ansatz.copy()
    measured_circuit.measure_all()
    transpiled_measured = deps["transpile"](measured_circuit, backend=backend)
    bound_circuit = transpiled_measured.assign_parameters(optimal_parameters)

    sampler = deps["SamplerV2"]()
    try:
        job = sampler.run([bound_circuit], shots=config.shots)
        result = job.result()
        counts = result[0].data.meas.get_counts()
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain-specific error
        raise QAOASolverError(f"Qiskit Aer simulator execution failed: {exc}") from exc

    return dict(counts)


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def _select_best_valid_solution(
    counts: Dict[str, int],
    canonical: _CanonicalQUBO,
    shots: int,
) -> Tuple[Optional[Dict[str, int]], Optional[float], Optional[float]]:
    """
    From measured bitstring counts, evaluate the ORIGINAL QUBO energy
    for every distinct measured bitstring, discard bitstrings that
    violate the exactly-one-phase constraint, and return the valid
    assignment with the lowest QUBO energy.

    Selection is by energy, never by measurement probability alone —
    e.g. a 40%-probability invalid or higher-energy bitstring never
    beats a 35%-probability lower-energy one.

    Returns:
        (best_assignment_or_None, best_energy_or_None, best_probability_or_None)
    """
    best_assignment: Optional[Dict[str, int]] = None
    best_energy: Optional[float] = None
    best_probability: Optional[float] = None

    for bitstring, count in counts.items():
        assignment = _bitstring_to_assignment(bitstring, canonical.variables)
        if not _is_exactly_one_hot(assignment):
            continue
        energy = _evaluate_qubo_energy(canonical, assignment)
        probability = count / shots
        if best_energy is None or energy < best_energy:
            best_assignment = assignment
            best_energy = energy
            best_probability = probability

    return best_assignment, best_energy, best_probability


def _fallback_best_valid_solution(
    canonical: _CanonicalQUBO,
) -> Tuple[Dict[str, int], float]:
    """
    Documented fallback used only when config.fallback_on_no_valid_solution
    is True and no measured bitstring was valid: evaluate the original
    QUBO energy directly over the exact one-hot assignments (one
    variable set to 1, all others 0) and return the best one. This
    bypasses quantum measurement entirely and is always marked via
    `used_fallback: True` in the returned result — never a silent
    substitution.
    """
    best_assignment: Optional[Dict[str, int]] = None
    best_energy: Optional[float] = None
    for active_var in canonical.variables:
        assignment = {var: (1 if var == active_var else 0) for var in canonical.variables}
        energy = _evaluate_qubo_energy(canonical, assignment)
        if best_energy is None or energy < best_energy:
            best_assignment = assignment
            best_energy = energy
    # With >=2 variables this is always populated.
    return best_assignment, best_energy  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def solve(
    qubo: Dict,
    junction_id: Optional[str] = None,
    candidate_roads: Optional[Sequence[str]] = None,
    config: Optional[QAOASolverConfig] = None,
) -> Dict:
    """
    Solve a QUBO produced by qubo_builder.py using QAOA on a local
    Qiskit Aer simulator, and return the selected phase.

    This function does NOT know or care which traffic-priority mode
    (NORMAL / HIGH_DENSITY) was used to construct the QUBO — it only
    converts the given QUBO to an Ising Hamiltonian, runs QAOA to
    minimize it, and re-evaluates the ORIGINAL QUBO energy of every
    measured bitstring to select the best constraint-valid solution.

    Args:
        qubo: The QUBO dict from qubo_builder.py (or an equivalent
            structure exposing candidate_phases/linear_terms/
            quadratic_terms/constant_offset, or a qubo_matrix).
        junction_id: Optional junction identifier; if omitted, taken
            from qubo["junction_id"] when present.
        candidate_roads: Optional legacy/back-compat interface
            parameter, consulted only if the QUBO itself does not
            already specify candidate_phases/linear_terms.
        config: QAOA solver configuration; defaults to
            QAOASolverConfig() if omitted.

    Returns:
        {
            "solver": "qaoa",
            "junction_id": ...,
            "selected_phase": "NS" or "EW",
            "binary_solution": [1, 0] or [0, 1],
            "best_energy": ...,
            "constraint_valid": True,
            "qaoa_depth": ...,
            "shots": ...,
            "backend": "local_aer_simulator",
            "optimizer": ...,
            "best_probability": ...,
            "measurement_counts": {...},
            "optimization_iterations": ...,
            "execution_time_ms": ...,
            "random_seed": ...,
            # present only if the documented fallback was triggered:
            "used_fallback": True,
        }

    Raises:
        QAOASolverError: for missing/empty/malformed QUBO input,
            invalid configuration, or missing Qiskit/Aer/SciPy.
        QAOAOptimizationError: if the classical parameter-optimization
            loop fails.
        NoValidQAOASolutionError: if no measured bitstring satisfies
            the exactly-one-phase constraint and no fallback is
            configured.
    """
    import time
    import numpy as np

    config = config or QAOASolverConfig()
    config.validate()

    if qubo is None:
        raise QAOASolverError("No QUBO was supplied (qubo is None).")
    resolved_junction_id = junction_id if junction_id is not None else qubo.get("junction_id")

    start = time.perf_counter()

    # 1-2. Parse QUBO, convert to Ising.
    canonical = _canonicalize_qubo(qubo, candidate_roads)
    h, j_coeffs, ising_constant = _qubo_to_ising(canonical)

    # Lazy Qiskit/Aer/SciPy import — only required once solve() actually runs.
    deps = _import_qiskit_dependencies()

    cost_operator = _build_cost_operator(canonical.variables, h, j_coeffs, deps["SparsePauliOp"])

    # 3-6. Build QAOA ansatz (cost operator + default mixer + initial state),
    # parameterized by (beta, gamma) across `qaoa_depth` layers.
    ansatz = deps["QAOAAnsatz"](cost_operator=cost_operator, reps=config.qaoa_depth)

    backend = deps["AerSimulator"](seed_simulator=config.random_seed)
    transpiled_ansatz = deps["transpile"](ansatz, backend=backend)

    rng = np.random.default_rng(config.random_seed)

    # 7-8. Optimize (beta, gamma) to minimize the QUBO objective's
    # expectation value under the cost Hamiltonian.
    optimal_parameters, optimal_objective_value, optimization_iterations = _optimize_qaoa_parameters(
        deps, transpiled_ansatz, cost_operator, ising_constant, config, rng
    )

    # 9-11. Execute the optimized circuit on the local Aer simulator and measure.
    counts = _sample_measurement_counts(deps, transpiled_ansatz, optimal_parameters, config, backend)

    # 12-14. Validate candidates and select the lowest-ORIGINAL-QUBO-energy valid solution.
    best_assignment, best_energy, best_probability = _select_best_valid_solution(
        counts, canonical, config.shots
    )

    used_fallback = False
    if best_assignment is None:
        if not config.fallback_on_no_valid_solution:
            raise NoValidQAOASolutionError(
                "No measured bitstring satisfied the exactly-one-phase constraint "
                f"(measurement_counts={counts}). Set "
                "QAOASolverConfig(fallback_on_no_valid_solution=True) to allow a "
                "documented classical fallback instead of raising."
            )
        best_assignment, best_energy = _fallback_best_valid_solution(canonical)
        best_probability = 0.0
        used_fallback = True

    constraint_valid = _is_exactly_one_hot(best_assignment)
    if not constraint_valid:
        # Unreachable given the selection/fallback logic above only ever
        # returns one-hot assignments, but validated explicitly per the
        # requirement to never silently return an invalid phase.
        raise NoValidQAOASolutionError(
            f"Best solution found violates the exactly-one-phase constraint: {best_assignment}"
        )

    # Section 40 requires every solver result to be validated, including
    # "returned energy == recomputed QUBO energy". best_energy already
    # comes from _evaluate_qubo_energy(), but this is asserted explicitly
    # -- against a fresh recompute -- so a future refactor that lets
    # best_energy drift away from the true QUBO energy fails loudly
    # instead of silently returning a wrong value.
    recomputed_energy = _evaluate_qubo_energy(canonical, best_assignment)
    if abs(recomputed_energy - best_energy) > 1e-6:
        raise NoValidQAOASolutionError(
            f"Returned energy ({best_energy}) does not match the recomputed "
            f"QUBO energy ({recomputed_energy}) for assignment {best_assignment}."
        )

    selected_phase = next(var for var, value in best_assignment.items() if value == 1)
    binary_solution = [best_assignment[var] for var in canonical.variables]

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    output = {
        "solver": "qaoa",
        "junction_id": resolved_junction_id,
        "selected_phase": selected_phase,
        "binary_solution": binary_solution,
        "best_energy": best_energy,
        "constraint_valid": constraint_valid,
        "qaoa_depth": config.qaoa_depth,
        "shots": config.shots,
        "backend": config.backend,
        "optimizer": config.optimizer,
        "best_probability": best_probability,
        "measurement_counts": counts,
        "optimization_iterations": optimization_iterations,
        "execution_time_ms": elapsed_ms,
        "random_seed": config.random_seed,
    }
    if used_fallback:
        output["used_fallback"] = True
    return output


# ---------------------------------------------------------------------------
# Example usage / smoke test (manual; not executed on import)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import itertools

    # --- Correctness check for the QUBO -> Ising conversion itself,
    # independent of Qiskit, verifying it preserves solution ordering. ---
    _canonical_check = _CanonicalQUBO(
        variables=["NS", "EW"],
        linear={"NS": -11.0, "EW": -10.0},
        quadratic={("NS", "EW"): 20.0},
        constant=10.0,
    )
    _h, _j, _c = _qubo_to_ising(_canonical_check)
    for _bits in itertools.product((0, 1), repeat=2):
        _x = dict(zip(_canonical_check.variables, _bits))
        _z = {v: (1 - 2 * _x[v]) for v in _canonical_check.variables}
        _qubo_e = _evaluate_qubo_energy(_canonical_check, _x)
        _ising_e = _c + sum(_h[v] * _z[v] for v in _canonical_check.variables) + sum(
            coeff * _z[i] * _z[j] for (i, j), coeff in _j.items()
        )
        assert abs(_qubo_e - _ising_e) < 1e-9, (_x, _qubo_e, _ising_e)
    print("QUBO<->Ising equivalence check passed.")

    # --- Full end-to-end run (requires qiskit + qiskit-aer installed). ---
    example_qubo = {
        "junction_id": "J1",
        "candidate_phases": ["NS", "EW"],
        "linear_terms": {"NS": -11.0, "EW": -10.0},
        "quadratic_terms": {"NS_EW": 20.0},
        "constant_offset": 10.0,
        "qubo_matrix": [[-11.0, 10.0], [10.0, -10.0]],
    }

    result = solve(example_qubo, config=QAOASolverConfig(qaoa_depth=1, shots=2000, random_seed=42))
    print(json.dumps(result, indent=2))
    assert result["selected_phase"] == "NS"
    assert result["binary_solution"] == [1, 0]
    assert result["constraint_valid"] is True

    # EW is more congested here -> EW should win.
    example_qubo_2 = {
        "linear_terms": {"NS": -5.0, "EW": -8.0},
        "quadratic_terms": {("NS", "EW"): 20.0},
        "constant_offset": 10.0,
    }
    result2 = solve(
        example_qubo_2,
        junction_id="J2",
        candidate_roads=["NS", "EW"],
        config=QAOASolverConfig(qaoa_depth=2, shots=2000, random_seed=7),
    )
    print(json.dumps(result2, indent=2))
    assert result2["selected_phase"] == "EW"
    assert result2["binary_solution"] == [0, 1]

    # Error handling checks.
    try:
        solve(None)
        raise AssertionError("Expected QAOASolverError for missing QUBO")
    except QAOASolverError:
        pass

    try:
        solve({})
        raise AssertionError("Expected QAOASolverError for empty QUBO")
    except QAOASolverError:
        pass

    try:
        solve(example_qubo, config=QAOASolverConfig(qaoa_depth=0))
        raise AssertionError("Expected QAOASolverError for invalid qaoa_depth")
    except QAOASolverError:
        pass

    try:
        solve(example_qubo, config=QAOASolverConfig(optimizer="not_a_real_optimizer"))
        raise AssertionError("Expected QAOASolverError for invalid optimizer")
    except QAOASolverError:
        pass

    print("All smoke tests passed.")