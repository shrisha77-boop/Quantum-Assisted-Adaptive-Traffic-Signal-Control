# Work to be done

# Receive:
# Current traffic metrics.
# Predicted traffic metrics.
# Available approaches.
# Construct the optimisation objective.
# Build the QUBO matrix.
# Return the QUBO.

"""
qubo_builder.py

QUBO Formulation Layer for the Adaptive Alternate Green Wave Traffic
Signal Coordination System.

Position in the architecture:

    ... -> Decision Engine -> qubo_builder.py (THIS FILE)
         -> QUBO Solver (QAOA / Classical) -> Signal Controller

This module ONLY formulates the optimisation problem. It does not read
TraCI, collect data, detect blockage/emergencies, calculate any traffic
metric, calculate green time, select a phase, control signals, or solve
anything. It receives already-filtered candidate roads and their metrics
from decision_engine.py and returns a solver-independent QUBO.

QUBO derivation used here (documented for traceability)
---------------------------------------------------------
Objective (linear in the binary variables):

    obj_i = waiting_weight  * normalized_waiting_time[i]
          + density_weight  * normalized_density[i]
          + queue_weight    * normalized_queue_length[i]

One-hot constraint "exactly one road selected", penalised quadratically:

    penalty * (sum_i x_i - 1)^2

Since x_i is binary, x_i^2 = x_i, so this constraint expands to:

    penalty * ( -sum_i x_i + 2 * sum_{i<j} x_i*x_j + 1 )

Combining the (purely linear) objective with the (linear + quadratic)
constraint gives, per candidate road i and each ordered pair (i, j):

    linear_terms[i]        = obj_i - penalty
    quadratic_terms[(i,j)] = 2 * penalty        for every i < j
    constant_offset        = penalty

This is the standard solver-independent QUBO dictionary form (as used by
libraries such as dimod): objective = sum_i linear_i*x_i
                                     + sum_{i<j} quadratic_ij*x_i*x_j
                                     + constant_offset
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModeWeights:
    """Weight set applied to the three normalized metrics in one traffic mode."""

    waiting: float
    density: float
    queue: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class QUBOConfig:
    """Centralised, tunable configuration for QUBO construction. No value
    used by QUBOBuilder is hard-coded outside of this class."""

    # --- Adaptive priority / hysteresis thresholds (on RAW queue storage
    # ratio, not the normalized value -- see module docstring) ---
    density_threshold: float = 0.65             # enters CONGESTED mode at/above this
    density_recovery_threshold: float = 0.60     # returns to NORMAL mode at/below this

    # --- Mode-specific weight sets ---
    normal_weights: ModeWeights = field(
        default_factory=lambda: ModeWeights(waiting=0.50, density=0.30, queue=0.20)
    )
    congested_weights: ModeWeights = field(
        default_factory=lambda: ModeWeights(waiting=0.30, density=0.50, queue=0.20)
    )

    # --- One-hot constraint penalty ---
    penalty_coefficient: float = 8.0

    # --- Normalization method shared by all three metrics ---
    # "minmax" scales each metric to [0, 1] across the current candidate
    # set; "zscore" standardises to zero mean / unit variance.
    normalization_method: str = "minmax"

    def __post_init__(self) -> None:
        if self.density_recovery_threshold > self.density_threshold:
            raise ValueError(
                "density_recovery_threshold must be <= density_threshold "
                "for hysteresis to behave correctly."
            )


# ---------------------------------------------------------------------------
# Metric normalization (deliberately separate from QUBO construction)
# ---------------------------------------------------------------------------

class MetricNormalizer:
    """
    Normalizes a list of raw metric values onto a common, comparable
    scale. Kept independent of QUBOBuilder so normalization strategy can
    be changed or extended without touching the QUBO assembly logic.
    """

    @staticmethod
    def normalize(values: List[float], method: str = "minmax") -> List[float]:
        if not values:
            return []

        if method == "minmax":
            lo, hi = min(values), max(values)
            spread = hi - lo
            if spread == 0:
                # All candidates equal on this metric -- no discriminative
                # signal, so contribute zero rather than risk a division
                # by zero or an arbitrary tie-break.
                return [0.0 for _ in values]
            return [(v - lo) / spread for v in values]

        if method == "zscore":
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance ** 0.5
            if std == 0:
                return [0.0 for _ in values]
            return [(v - mean) / std for v in values]

        raise ValueError(f"Unsupported normalization method: {method!r}")


# ---------------------------------------------------------------------------
# QUBO Builder
# ---------------------------------------------------------------------------

class QUBOBuilder:
    """
    Builds a solver-independent QUBO formulation for selecting exactly
    one candidate road/phase at a junction, using adaptive, hysteresis-
    controlled weighting between waiting time, density, and queue length.

    One instance should persist across simulation steps (per junction, or
    shared, since the only state it carries is the current traffic mode
    for hysteresis purposes).
    """

    def __init__(self, config: QUBOConfig = None) -> None:
        self.config = config or QUBOConfig()
        # Hysteresis state, persists across build() calls -- keyed by
        # junction_id. The corridor has MULTIPLE junctions and a single
        # QUBOBuilder instance is meant to be shared across all of them
        # (see class docstring), so the mode must be tracked per junction.
        # A single scalar self._mode would let one junction's congestion
        # state leak into every other junction's weight selection.
        self._junction_modes: Dict[Any, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, decision_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construct the QUBO for one junction's decision.

        Parameters
        ----------
        decision_input : dict with at least:
            "junction_id"          : str
            "candidate_roads"      : List[str]
            "waiting_time"         : Dict[road_id, float]
            "queue_storage_ratio"  : Dict[road_id, float]   (RQ, HCM-style)
            "queue_length"         : Dict[road_id, float]

        Returns
        -------
        Structured dict matching the module's OUTPUT contract.
        """
        junction_id = decision_input.get("junction_id")
        candidate_roads: List[str] = list(decision_input.get("candidate_roads", []))

        if not candidate_roads:
            # Nothing to optimise over -- return a well-formed, empty QUBO
            # rather than let the caller crash on missing keys.
            return self._empty_result(junction_id)

        waiting_raw = self._collect(decision_input.get("waiting_time", {}), candidate_roads)
        density_raw = self._collect(decision_input.get("queue_storage_ratio", {}), candidate_roads)
        queue_raw = self._collect(decision_input.get("queue_length", {}), candidate_roads)

        norm_method = self.config.normalization_method
        norm_wait = MetricNormalizer.normalize(waiting_raw, norm_method)
        norm_density = MetricNormalizer.normalize(density_raw, norm_method)
        norm_queue = MetricNormalizer.normalize(queue_raw, norm_method)

        # Mode decision uses the RAW density (queue storage ratio), never
        # the normalized value -- the 0.65/0.60 thresholds are defined on
        # the physical RQ scale.
        max_density = max(density_raw) if density_raw else 0.0
        mode = self._update_mode(junction_id, max_density)
        weights = self._weights_for_mode(mode)

        decision_variables = {road: idx for idx, road in enumerate(candidate_roads)}

        objective_coeffs = self._build_objective_coefficients(
            candidate_roads, norm_wait, norm_density, norm_queue, weights
        )
        linear_terms, quadratic_terms, constant_offset = self._build_qubo_terms(
            candidate_roads, objective_coeffs, self.config.penalty_coefficient
        )
        qubo_matrix = self._build_matrix(decision_variables, linear_terms, quadratic_terms)

        return {
            "junction_id": junction_id,
            "candidate_roads": candidate_roads,
            "decision_variables": decision_variables,
            "traffic_mode": mode,
            "weights": weights.as_dict(),
            "qubo_matrix": qubo_matrix,
            "linear_terms": linear_terms,
            "quadratic_terms": quadratic_terms,
            "constant_offset": constant_offset,
        }

    def current_mode(self, junction_id: Any = None) -> Any:
        """
        Return the current hysteresis-controlled traffic mode.

        If junction_id is given, returns that junction's mode (defaulting
        to "NORMAL" if it hasn't been seen yet). If junction_id is
        omitted, returns the full {junction_id: mode} mapping for all
        junctions tracked so far.
        """
        if junction_id is None:
            return dict(self._junction_modes)
        return self._junction_modes.get(junction_id, "NORMAL")

    def reset_mode(self, junction_id: Any = None, mode: str = "NORMAL") -> None:
        """
        Reset hysteresis state, e.g. when starting a new simulation run.

        If junction_id is given, resets only that junction. If omitted,
        resets every junction currently tracked back to `mode`.
        """
        if mode not in ("NORMAL", "CONGESTED"):
            raise ValueError("mode must be 'NORMAL' or 'CONGESTED'")
        if junction_id is None:
            for jid in list(self._junction_modes.keys()):
                self._junction_modes[jid] = mode
        else:
            self._junction_modes[junction_id] = mode

    # ------------------------------------------------------------------
    # Internal: adaptive priority / hysteresis
    # ------------------------------------------------------------------

    def _update_mode(self, junction_id: Any, max_density: float) -> str:
        """
        Advance the persistent hysteresis state machine FOR THIS JUNCTION.
        Stays in the current mode until the OPPOSITE threshold is crossed,
        preventing rapid oscillation when density hovers near a single
        threshold value. Each junction's mode is independent, since one
        junction being congested must not push a different, uncongested
        junction into CONGESTED weighting.
        """
        cfg = self.config
        mode = self._junction_modes.get(junction_id, "NORMAL")
        if mode == "NORMAL":
            if max_density >= cfg.density_threshold:
                mode = "CONGESTED"
        else:  # currently CONGESTED
            if max_density <= cfg.density_recovery_threshold:
                mode = "NORMAL"
        self._junction_modes[junction_id] = mode
        return mode

    def _weights_for_mode(self, mode: str) -> ModeWeights:
        return self.config.congested_weights if mode == "CONGESTED" else self.config.normal_weights

    # ------------------------------------------------------------------
    # Internal: objective construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_objective_coefficients(
        roads: List[str],
        norm_wait: List[float],
        norm_density: List[float],
        norm_queue: List[float],
        weights: ModeWeights,
    ) -> Dict[str, float]:
        """Per-road weighted sum of normalized metrics -- negated so energy minimization selects high-demand phase."""
        return {
            road: -(
                weights.waiting * norm_wait[i]
                + weights.density * norm_density[i]
                + weights.queue * norm_queue[i]
            )
            for i, road in enumerate(roads)
        }

    # ------------------------------------------------------------------
    # Internal: constraint + final QUBO term assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _build_qubo_terms(
        roads: List[str],
        objective_coeffs: Dict[str, float],
        penalty: float,
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], float]:
        """
        Combine the linear objective with the quadratic one-hot penalty.
        See the module docstring for the full derivation of these three
        expressions.
        """
        linear_terms: Dict[str, float] = {
            road: objective_coeffs[road] - penalty for road in roads
        }

        quadratic_terms: Dict[Tuple[str, str], float] = {}
        for i in range(len(roads)):
            for j in range(i + 1, len(roads)):
                quadratic_terms[(roads[i], roads[j])] = 2.0 * penalty

        constant_offset = penalty
        return linear_terms, quadratic_terms, constant_offset

    @staticmethod
    def _build_matrix(
        decision_variables: Dict[str, int],
        linear_terms: Dict[str, float],
        quadratic_terms: Dict[Tuple[str, str], float],
    ) -> List[List[float]]:
        """
        Dense n x n matrix view of the same QUBO, for solvers/tools that
        expect x^T Q x directly instead of the linear/quadratic dict form.

        NOTE on convention: because x^T Q x double-counts every
        off-diagonal pair (Q_ij and Q_ji both contribute), each
        off-diagonal entry here is HALF of the corresponding
        `quadratic_terms` value. The diagonal carries the full linear
        term (since x_i^2 = x_i for binary variables). The canonical,
        non-redundant representation is `linear_terms` + `quadratic_terms`
        -- use the matrix only where a literal x^T Q x form is required.
        """
        n = len(decision_variables)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]

        for road, idx in decision_variables.items():
            matrix[idx][idx] = linear_terms[road]

        for (road_a, road_b), coeff in quadratic_terms.items():
            i, j = decision_variables[road_a], decision_variables[road_b]
            half = coeff / 2.0
            matrix[i][j] = half
            matrix[j][i] = half

        return matrix

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect(metric_map: Dict[str, float], roads: List[str]) -> List[float]:
        """Pull one metric's values out in candidate-road order, defaulting missing entries to 0.0."""
        return [float(metric_map.get(road, 0.0) or 0.0) for road in roads]

    def _empty_result(self, junction_id: Any) -> Dict[str, Any]:
        mode = self._junction_modes.get(junction_id, "NORMAL")
        return {
            "junction_id": junction_id,
            "candidate_roads": [],
            "decision_variables": {},
            "traffic_mode": mode,
            "weights": self._weights_for_mode(mode).as_dict(),
            "qubo_matrix": [],
            "linear_terms": {},
            "quadratic_terms": {},
            "constant_offset": 0.0,
        }