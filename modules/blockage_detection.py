# 1. Receive raw vehicle data and traffic metrics.
# 2. Monitor each incoming road/approach.
# 3. Check blockage conditions:
#    - Queue length exceeds threshold.
#    - Average waiting time exceeds threshold.
#    - Average speed falls below threshold.
# 4. Confirm blockage only if the condition persists for a specified duration.
# 5. Mark the road as BLOCKED.
# 6. Monitor the blocked road continuously.
# 7. Detect when the blockage is cleared.
# 8. Send the road status to the next module.
"""
blockage_detection.py

Blockage Detection Layer for the Adaptive Alternate Green Wave
Traffic Signal Management System.

Position in the architecture:

    SUMO -> TraCI API -> traci_interface.py -> data_collection.py
         -> blockage_detection.py (THIS FILE) -> decision_engine.py
         -> QUBO Builder -> QAOA Solver -> signal_controller.py

This module does NOT talk to TraCI. It only consumes the structured
per-lane data produced by data_collection.py, and only produces a
structured blockage-status dictionary for decision_engine.py to consume.

Formulas used (see project research notes for full derivations/citations):

    N_max   = lane_length / (avg_vehicle_length + jam_spacing)
              (physical jam-density-based queue capacity of the lane)

    RQ      = current_queue_length / N_max
              (HCM-style Queue Storage Ratio; RQ > 1.0 => spillback)

    speed_ratio = mean_lane_speed / max_lane_speed
              (relative-speed congestion indicator; road-independent)

A lane is flagged as a *candidate* blockage only when all three
conditions hold simultaneously, and it is only declared BLOCKED once
that candidate condition has persisted for a configurable number of
consecutive simulation steps (temporal persistence / debouncing), and
similarly only cleared after the cleared condition has persisted for
that many steps. This avoids state oscillation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BlockageConfig:
    """
    Centralised, tunable configuration for the blockage detector.
    Keep every numerical constant used by the algorithm here so the
    module never hardcodes a value inline.
    """

    # Jam spacing: standstill gap between the front bumpers of two
    # stopped vehicles, in metres. Typical published range: 1-2 m.
    jam_spacing_m: float = 1.5

    # Fallback average vehicle length (metres) used only when a lane
    # currently has no vehicles on it to measure from.
    default_vehicle_length_m: float = 5.0

    # HCM-style thresholds (see module docstring for the formulas).
    queue_storage_ratio_threshold: float = 1.0     # RQ > 1.0 => spillback
    waiting_time_threshold_s: float = 80.0          # HCM LOS E/F boundary
    speed_ratio_threshold: float = 0.30             # relative speed drop

    # Temporal persistence: number of consecutive simulation steps the
    # condition must hold before a state transition (CLEAR<->BLOCKED)
    # is confirmed. Same duration is used for both directions unless
    # clearance_persistence_steps is explicitly overridden.
#     The lane must satisfy all blockage conditions for 5 consecutive checks before it becomes BLOCKED.
# After it is blocked, the lane must remain normal/clear for 5 consecutive checks before it becomes CLEAR again.
    persistence_steps: int = 5
    clearance_persistence_steps: int = 5



# ---------------------------------------------------------------------------
# Internal per-lane state
# ---------------------------------------------------------------------------

@dataclass
class _LaneState:
    """Internal bookkeeping kept per monitored lane between update() calls."""

    status: str = "CLEAR"          # "CLEAR" or "BLOCKED"
    block_counter: int = 0         # consecutive steps the block condition has held
    clear_counter: int = 0         # consecutive steps the clear condition has held


# ---------------------------------------------------------------------------
# Blockage Detection
# ---------------------------------------------------------------------------

class BlockageDetection:
    """
    Detects and tracks road blockages on every incoming lane of every
    monitored junction, using only data supplied by the Data Collection
    Layer (never TraCI directly).

    Usage:
        detector = BlockageDetection()
        results = detector.update(lane_data_from_data_collection, sim_time)
        # results: Dict[lane_id, dict] -- see _build_result() for schema.
    """

    def __init__(self, config: Optional[BlockageConfig] = None) -> None:
        self.config: BlockageConfig = config or BlockageConfig()
        self._lane_states: Dict[str, _LaneState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        lane_data: Dict[str, Dict[str, Any]],
        sim_time: float = 0.0,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process one simulation step's worth of lane data and return the
        current blockage status for every lane provided.

        Parameters
        ----------
        lane_data : mapping of lane_id -> raw lane data, where each raw
            lane data dict is expected to contain at least:
                "junction_id"          : str
                "lane_length"          : float (metres)
                "max_speed"            : float (m/s)
                "mean_speed"           : float (m/s)
                "halting_number"       : int
                "vehicle_ids"          : List[str]
                "vehicle_lengths"      : Dict[vehicle_id, float]  (metres)
                "vehicle_waiting_times": Dict[vehicle_id, float]  (seconds)
            Missing optional keys are handled safely with sane fallbacks.
        sim_time : current simulation time, echoed back in each result
            for downstream logging/traceability.

        Returns
        -------
        Dict[lane_id, result_dict] -- one entry per lane in `lane_data`.
        """
        results: Dict[str, Dict[str, Any]] = {}

        for lane_id, raw in lane_data.items():
            metrics = self._compute_metrics(raw) #Calculate traffic values:
            candidate_blocked = self._evaluate_conditions(metrics) #Check whether the lane currently looks blocked
            state = self._advance_state(lane_id, candidate_blocked) #Update the remembered lane status
            results[lane_id] = self._build_result(  #Save the final result:
                lane_id=lane_id,
                junction_id=raw.get("junction_id"),
                raw=raw,
                metrics=metrics,
                candidate_blocked=candidate_blocked,
                state=state,
                sim_time=sim_time,
            )

        return results

    def get_status(self, lane_id: str) -> str: #This function tells you the current saved status of one lane.
        """Return the last known confirmed status ('CLEAR' or 'BLOCKED') for a lane."""
        state = self._lane_states.get(lane_id)
        return state.status if state else "CLEAR"

    def reset(self, lane_id: Optional[str] = None) -> None:
        """
        Reset internal persistence/state tracking.
        If lane_id is given, resets only that lane; otherwise resets all lanes.
        Useful when restarting a simulation run.
        """
        if lane_id is None:
            self._lane_states.clear()
        else:
            self._lane_states.pop(lane_id, None)

    # ------------------------------------------------------------------
    # Internal: metric calculation (Steps 2-5 of the workflow)
    # ------------------------------------------------------------------

    def _compute_metrics(self, raw: Dict[str, Any]) -> Dict[str, float]:
        """Compute N_max, RQ, average waiting time, and speed_ratio for one lane."""
        lane_length = float(raw.get("lane_length", 0.0) or 0.0)
        max_speed = float(raw.get("max_speed", 0.0) or 0.0)
        mean_speed = float(raw.get("mean_speed", 0.0) or 0.0)
        halting_number = float(raw.get("halting_number", 0) or 0)

        vehicle_lengths: Dict[str, float] = raw.get("vehicle_lengths") or {}
        vehicle_waiting_times: Dict[str, float] = raw.get("vehicle_waiting_times") or {}
        vehicle_ids: List[str] = raw.get("vehicle_ids") or []

        # --- Step 2: physical queue capacity (N_max) ---
        avg_vehicle_length = (
            (sum(vehicle_lengths.values()) / len(vehicle_lengths))
            if vehicle_lengths
            else self.config.default_vehicle_length_m
        )
        denom = avg_vehicle_length + self.config.jam_spacing_m
        n_max = (lane_length / denom) if denom > 0 and lane_length > 0 else 0.0

        # --- Step 3: queue storage ratio (RQ) ---
        rq = self._safe_divide(halting_number, n_max)

        # --- Step 4: average waiting time ---
        if vehicle_ids and vehicle_waiting_times:
            wait_values = [
                vehicle_waiting_times.get(vid, 0.0) for vid in vehicle_ids
            ]
            avg_wait = sum(wait_values) / len(wait_values) if wait_values else 0.0
        else:
            avg_wait = 0.0

        # --- Step 5: relative speed ratio ---
        # If max_speed is unavailable/zero, we cannot form a meaningful
        # ratio; default to 1.0 (free-flow) rather than risk a false
        # blockage flag from bad/missing data.
        speed_ratio = self._safe_divide(mean_speed, max_speed, default=1.0)

        return {
            "n_max": n_max,
            "rq": rq,
            "avg_wait": avg_wait,
            "speed_ratio": speed_ratio,
            "avg_vehicle_length": avg_vehicle_length,
            "halting_number": halting_number,
        }

    # ------------------------------------------------------------------
    # Internal: condition evaluation (Step 6)
    # ------------------------------------------------------------------

    def _evaluate_conditions(self, metrics: Dict[str, float]) -> bool:
        """Return True if all three blockage conditions currently hold."""
        cfg = self.config
        return (
            metrics["rq"] > cfg.queue_storage_ratio_threshold
            and metrics["avg_wait"] > cfg.waiting_time_threshold_s
            and metrics["speed_ratio"] < cfg.speed_ratio_threshold
        )

    def _failing_reasons(self, metrics: Dict[str, float]) -> List[str]:
        """List which individual raw conditions are currently true (diagnostics)."""
        cfg = self.config
        reasons: List[str] = []
        if metrics["rq"] > cfg.queue_storage_ratio_threshold:
            reasons.append(
                f"queue_storage_ratio={metrics['rq']:.2f} > {cfg.queue_storage_ratio_threshold}"
            )
        if metrics["avg_wait"] > cfg.waiting_time_threshold_s:
            reasons.append(
                f"average_waiting_time={metrics['avg_wait']:.1f}s > {cfg.waiting_time_threshold_s}s"
            )
        if metrics["speed_ratio"] < cfg.speed_ratio_threshold:
            reasons.append(
                f"speed_ratio={metrics['speed_ratio']:.2f} < {cfg.speed_ratio_threshold}"
            )
        return reasons

    # ------------------------------------------------------------------
    # Internal: temporal persistence / state machine (Steps 7-9)
    # ------------------------------------------------------------------

    def _advance_state(self, lane_id: str, candidate_blocked: bool) -> _LaneState:
        """
        Advance the per-lane persistence counters and confirm a state
        transition only once the required number of consecutive steps
        has been observed, in either direction. Prevents oscillation.
        """
        state = self._lane_states.setdefault(lane_id, _LaneState())
        cfg = self.config

        if state.status == "CLEAR":
            if candidate_blocked:
                state.block_counter += 1
                state.clear_counter = 0
            else:
                state.block_counter = 0

            if state.block_counter >= cfg.persistence_steps:
                state.status = "BLOCKED"
                state.block_counter = 0
                state.clear_counter = 0

        else:  # state.status == "BLOCKED"
            if not candidate_blocked:
                state.clear_counter += 1
                state.block_counter = 0
            else:
                state.clear_counter = 0

            if state.clear_counter >= cfg.clearance_persistence_steps:
                state.status = "CLEAR"
                state.block_counter = 0
                state.clear_counter = 0

        return state

    # ------------------------------------------------------------------
    # Internal: output assembly (Step 10)
    # ------------------------------------------------------------------

    def _build_result(
        self,
        lane_id: str,
        junction_id: Optional[str],
        raw: Dict[str, Any],
        metrics: Dict[str, float],
        candidate_blocked: bool,
        state: _LaneState,
        sim_time: float,
    ) -> Dict[str, Any]:
        active_counter = (
            state.block_counter if state.status == "CLEAR" else state.clear_counter
        )
        return {
            "lane_id": lane_id,
            "junction_id": junction_id,
            "current_queue_length": metrics["halting_number"],
            "max_queue_capacity": metrics["n_max"],
            "queue_storage_ratio": metrics["rq"],
            "average_waiting_time": metrics["avg_wait"],
            "speed_ratio": metrics["speed_ratio"],
            "blockage_counter": active_counter,
            "status": state.status,
            "candidate_blocked_this_step": candidate_blocked,
            "reasons": self._failing_reasons(metrics),
            "sim_time": sim_time,
            # Alias for consumers (road_isolation_manager.py) that key off
            # "timestamp" rather than "sim_time" -- kept in addition to
            # "sim_time" (not a replacement) so no existing consumer of the
            # original field name breaks.
            "timestamp": sim_time,
        }

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Division that never raises ZeroDivisionError; returns `default` instead."""
        if denominator == 0:
            return default
        return numerator / denominator