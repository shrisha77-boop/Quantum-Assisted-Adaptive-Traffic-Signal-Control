"""Top-level orchestration for the Bengaluru adaptive traffic controller.

Architecture
------------
This file coordinates the existing pipeline modules in order:

    SUMO / TraCI
         ↓
    Data Collection          (modules/data_collection.py)
         ↓
    Traffic Metrics          (modules/traffic_metric_calculation.py)
         ↓
    Blockage Detection       (modules/blockage_detection.py)
         ↓
    Road Isolation           (modules/road_isolation_manager.py)
         ↓
    Emergency Detection      (modules/emergency_preemption.py)
         ↓
    Predictor                (modules/predictor.py)
         ↓
    Decision Engine          (modules/decision_engine.py)
         ↓
    [QUBO Builder + Solver Manager run inside DecisionEngine.decide()]
         ↓
    Signal Controller        (modules/signal_controller.py)
         ↓
    SUMO

Threading contract
------------------
All live TraCI calls are made ONLY on the main simulation thread:

    Main thread owns:
        traci_api.step()                  ← simulationStep()
        traci_api.collect_step_data()     ← all TraCI getters
        data_collection.update(snapshot)  ← snapshot committed to shared state
        signal_controller.apply(...)      ← trafficlight.setRedYellowGreenState

    Background decision thread reads:
        data_collection.*                 ← safe: read-only snapshot already committed
        lane_data cache                   ← safe: built once on main thread at startup

    Background decision thread NEVER calls:
        traci.*                           ← prohibited
        traci_api.*                       ← prohibited

Static information (lane directions, lengths, signal groups) is cached
on the main thread during _initialize_runtime_modules() and then used
read-only by the decision thread.

Solver policy
-------------
Exactly one solver executes per decision cycle.  Solver selection is
delegated entirely to solver_manager.py / decision_engine.py:

    adaptive  →  QAOA for high traffic (N > HIGH_TRAFFIC_THRESHOLD)
                 Simulated Annealing for low/normal traffic

    simulated_annealing  →  always Simulated Annealing
    qaoa                 →  always QAOA

The following are explicitly NOT present in this file:
    - classical solver
    - control_solver
    - "all" multi-solver/benchmarking mode
    - solver comparison
    - solver fallback (a failing solver returns its failure to the caller)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import config
from scenario import scenario_config
from scenario.signals import signal_config

logger = logging.getLogger("main_controller")


class MainControllerError(RuntimeError):
    """Fatal controller/integration error."""


# ---------------------------------------------------------------------------
# Pure helpers — no TraCI calls; safe to call from any thread
# ---------------------------------------------------------------------------

def _direction_from_angle(angle: float) -> str:
    """Convert a SUMO heading angle (degrees, 0=North, clockwise) to N/E/S/W."""
    angle = float(angle) % 360.0
    if angle < 45 or angle >= 315:
        return "N"
    if angle < 135:
        return "E"
    if angle < 225:
        return "S"
    return "W"


def _lane_heading(traci_module, lane_id: str) -> float:
    """Compute a lane's compass heading from its shape (main-thread only)."""
    try:
        shape = traci_module.lane.getShape(lane_id)
        if shape and len(shape) >= 2:
            (x1, y1), (x2, y2) = shape[0], shape[-1]
            return math.degrees(math.atan2(x2 - x1, y2 - y1)) % 360.0
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class MainController:
    """Owns the simulation lifecycle and connects all project modules.

    Parameters
    ----------
    sumo_config : path to the .sumocfg file.
    use_gui : launch sumo-gui instead of headless sumo.
    decision_interval : seconds between decision cycles.
    solver : "adaptive" | "simulated_annealing" | "qaoa"
             Delegated to SolverManager; see solver_manager.py for policy.
    output_dir : directory where decision_results.jsonl is written.
    seed : optional SUMO random seed.
    """

    def __init__(
        self,
        sumo_config: str = scenario_config.SUMO_CONFIG,
        use_gui: bool = False,
        decision_interval: float = 30.0,
        solver: str = "adaptive",
        output_dir: str = "simulation/results",
        seed: Optional[int] = None,
    ) -> None:
        if decision_interval <= 0:
            raise MainControllerError("decision_interval must be positive.")
        valid_solvers = {"simulated_annealing", "qaoa", "adaptive"}
        if solver not in valid_solvers:
            raise MainControllerError(
                f"Unsupported solver: {solver!r}. Expected one of {sorted(valid_solvers)}."
            )

        self.sumo_config = sumo_config
        self.use_gui = use_gui
        self.decision_interval = decision_interval
        self.solver_name = solver
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Runtime module instances (populated by _load_modules /
        # _initialize_runtime_modules on the main thread).
        self.traci_api = None
        self.traci = None                     # the live `traci` module
        self.data_collection = None

        # Per-junction module instances
        self.metric_calculators: Dict[str, Any] = {}
        self.predictors: Dict[str, Any] = {}
        self.blockage_detectors: Dict[str, Any] = {}
        self.isolation_managers: Dict[str, Any] = {}
        self.emergency_detectors: Dict[str, Any] = {}
        self.decision_engines: Dict[str, Any] = {}
        self.signal_controllers: Dict[str, Any] = {}

        # Per-junction static geometry cache (built on main thread at startup;
        # read-only thereafter — safe to read from the decision thread).
        self.junction_runtime: Dict[str, Dict[str, Any]] = {}

        # Shared decision state (written by decision thread, read by main
        # thread to feed signal_controller.apply() each simulation step).
        self.latest_decisions: Dict[str, Any] = {}
        self._decision_lock = threading.Lock()

        self.next_decision_time = 0.0
        self.decision_count = 0
        self.results_file = self.output_dir / "decision_results.jsonl"

        # Background decision thread tracking
        self._decisions_in_flight = False
        self._decision_threads: List[threading.Thread] = []

    # ------------------------------------------------------------------
    # Module construction (main thread only)
    # ------------------------------------------------------------------

    def _load_modules(self) -> None:
        """Import and instantiate TraCI interface and data-collection layer."""
        from modules.traci_interface import TraCIConfig, TraCIInterface
        from modules.data_collection import DataCollectionLayer

        self.traci_api = TraCIInterface(
            TraCIConfig(
                sumocfg_path=self.sumo_config,
                use_gui=self.use_gui,
                step_length=config.SUMO_STEP_LENGTH,
                extra_sumo_args=(
                    ["--seed", str(self.seed)] if self.seed is not None else []
                ),
                # This controller reads only snapshot["vehicles"] and
                # snapshot["lanes"] per step.  Edges, junctions, traffic-light
                # snapshots, and turning movements are not consumed per-step
                # here (junction/TLS info is cached at startup via direct TraCI
                # calls on the main thread).  Disabling unused domains avoids
                # unnecessary per-step TraCI round-trips on large networks.
                collect_edges=False,
                collect_junctions=False,
                collect_traffic_lights=False,
                collect_turning_movements=False,
            )
        )
        self.data_collection = DataCollectionLayer()

    def _initialize_runtime_modules(self) -> None:
        """Build per-junction module instances.

        All TraCI calls in this method run on the main thread BEFORE the
        simulation loop starts — this is the only place static geometry
        (lane directions, lengths, signal groups) is read from TraCI.
        The resulting caches are stored in self.junction_runtime and used
        read-only afterwards, including from the background decision thread.
        """
        from modules.traffic_metric_calculation import TrafficMetricCalculator
        from modules.predictor import Predictor
        from modules.blockage_detection import BlockageDetection
        from modules.road_isolation_manager import RoadIsolationManager
        from modules.emergency_preemption import EmergencyVehicleDetection
        from modules.qubo_builder import QUBOBuilder
        from modules.qubo_solver.solver_manager import SolverManager, SolverManagerConfig
        from modules.green_time_calculator import GreenTimeCalculator
        from modules.signal_controller import SignalController
        from modules.decision_engine import DecisionEngine

        # One SolverManager shared across all junctions.
        # SolverManagerConfig only accepts: solver, simulated_annealing, qaoa.
        # The adaptive solver-selection policy (QAOA vs SA) is implemented
        # inside solver_manager.py based on vehicle_count — not here.
        self.solver_manager = SolverManager(
            SolverManagerConfig(solver=self.solver_name)
        )

        for junction_id, jcfg in signal_config.JUNCTIONS.items():
            candidate_roads = jcfg["candidate_roads"]
            phase_map = self._build_phase_map(candidate_roads)
            road_lengths = self._road_lengths_for_edges(candidate_roads)
            lane_data = self._lane_static_data(candidate_roads)
            signal_groups = self._build_signal_groups(jcfg["tls_id"])

            self.junction_runtime[junction_id] = {
                "tls_id": jcfg["tls_id"],
                "incoming_edges": set(candidate_roads),
                "phase_map": phase_map,
                # lane_id -> {"direction": str, "length": float}
                # Built once on the main thread; safe to read from any thread.
                "lane_data": lane_data,
            }

            self.metric_calculators[junction_id] = TrafficMetricCalculator(
                road_lengths=road_lengths
            )
            self.predictors[junction_id] = Predictor()
            self.blockage_detectors[junction_id] = BlockageDetection()
            self.isolation_managers[junction_id] = RoadIsolationManager(junction_id)
            self.emergency_detectors[junction_id] = EmergencyVehicleDetection(
                junction_id=junction_id,
                phase_map=phase_map,
            )
            self.decision_engines[junction_id] = DecisionEngine(
                qubo_builder=QUBOBuilder(),
                solver_manager=self.solver_manager,
                green_time_calculator=GreenTimeCalculator(),
            )

            # Synthesize conflict-free signal phase mapping for corridor junctions:
            # All links belonging to incoming NS edges get 'G' during NS phase.
            # All links belonging to incoming EW edges get 'G' during EW phase.
            controlled_links = self.traci.trafficlight.getControlledLinks(jcfg["tls_id"])
            num_links = len(controlled_links)
            ns_green = ["r"] * num_links
            ns_yellow = ["r"] * num_links
            ew_green = ["r"] * num_links
            ew_yellow = ["r"] * num_links
            all_red = "r" * num_links

            for idx, link_group in enumerate(controlled_links):
                if not link_group or not link_group[0]:
                    continue
                lane = link_group[0][0]
                edge = self.traci.lane.getEdgeID(lane)
                phase = phase_map.get(edge)
                if not phase:
                    d = self._edge_direction(edge)
                    phase = "NS" if d in {"N", "S"} else "EW"

                if phase == "NS":
                    ns_green[idx] = "G"
                    ns_yellow[idx] = "y"
                elif phase == "EW":
                    ew_green[idx] = "G"
                    ew_yellow[idx] = "y"

            phase_state_map = {
                "NS": {
                    "green": "".join(ns_green),
                    "yellow": "".join(ns_yellow),
                    "all_red": all_red,
                },
                "EW": {
                    "green": "".join(ew_green),
                    "yellow": "".join(ew_yellow),
                    "all_red": all_red,
                },
            }

            self.signal_controllers[junction_id] = SignalController(
                tls_id=jcfg["tls_id"],
                direction_signal_groups=signal_groups,
                traci_module=self.traci,
                phase_state_map=phase_state_map,
            )

        # Pre-compute monitored edges and lanes across all junctions.
        # This allows fast step snapshots without scanning the full network.
        self.monitored_edges = set()
        for jcfg in signal_config.JUNCTIONS.values():
            self.monitored_edges.update(jcfg["candidate_roads"])

        self.monitored_lanes = []
        for edge in self.monitored_edges:
            try:
                num_lanes = self.traci.edge.getLaneNumber(edge)
                for i in range(num_lanes):
                    self.monitored_lanes.append(f"{edge}_{i}")
            except Exception:
                pass

    def _get_fast_vehicle_data(self, vehicle_id: str) -> Dict[str, Any]:
        """Fetch only the required vehicle fields needed by downstream modules."""
        v = self.traci.vehicle
        try:
            return {
                "vehicle_id": vehicle_id,
                "type_id": v.getTypeID(vehicle_id),
                "vehicle_class": v.getVehicleClass(vehicle_id),
                "edge_id": v.getRoadID(vehicle_id),
                "lane_id": v.getLaneID(vehicle_id),
                "lane_position": float(v.getLanePosition(vehicle_id)),
                "position_xy": v.getPosition(vehicle_id),
                "speed": float(v.getSpeed(vehicle_id)),
                "max_speed": float(v.getMaxSpeed(vehicle_id)),
                "allowed_speed": float(v.getAllowedSpeed(vehicle_id)),
                "acceleration": float(v.getAcceleration(vehicle_id)),
                "angle": float(v.getAngle(vehicle_id)),
                "waiting_time": float(v.getWaitingTime(vehicle_id)),
                "accumulated_waiting_time": float(v.getAccumulatedWaitingTime(vehicle_id)),
                "length": float(v.getLength(vehicle_id)),
                "speed_factor": float(v.getSpeedFactor(vehicle_id)),
            }
        except Exception:
            return {}

    def _collect_fast_snapshot(self) -> Dict[str, Any]:
        """Collect raw SUMO state only for the monitored edges/lanes/vehicles."""
        # 1. Get simulation-level data
        sim_data = self.traci_api.get_simulation_data()

        # 2. Get lane data for monitored lanes only
        lanes = []
        vehicle_ids = set()
        for lane_id in self.monitored_lanes:
            lane_info = self.traci_api.get_lane_data(lane_id)
            if isinstance(lane_info, dict):
                lanes.append(lane_info)
                vids = lane_info.get("vehicle_ids", []) or []
                vehicle_ids.update(vids)

        # 3. Get vehicle data for monitored vehicles only
        vehicles = []
        for vid in vehicle_ids:
            v_info = self._get_fast_vehicle_data(vid)
            if v_info and v_info.get("vehicle_id"):
                vehicles.append(v_info)

        return {
            "simulation": sim_data,
            "vehicles": vehicles,
            "lanes": lanes,
            "edges": [],
            "junctions": [],
            "traffic_lights": [],
            "turning_movements": [],
        }

    # ------------------------------------------------------------------
    # Static geometry helpers — main thread only (called during init)
    # ------------------------------------------------------------------

    def _edge_direction(self, edge_id: str) -> str:
        """Return the compass direction (N/E/S/W) of a SUMO edge.

        Uses the heading of the first lane on that edge.  TraCI call —
        must be called on the main thread.
        """
        try:
            # The first lane of an edge in SUMO is always edge_id_0.
            lane_id = f"{edge_id}_0"
            return _direction_from_angle(
                _lane_heading(self.traci, lane_id)
            )
        except Exception:
            pass
        return "N"

    def _road_lengths_for_edges(
        self, edges: List[str]
    ) -> Dict[str, float]:
        """Average lane lengths per compass direction for the given edges.

        TraCI call — must be called on the main thread.
        """
        totals: Dict[str, List[float]] = {d: [] for d in config.DIRECTIONS}
        try:
            for edge in edges:
                direction = self._edge_direction(edge)
                num_lanes = self.traci.edge.getLaneNumber(edge)
                for i in range(num_lanes):
                    lane_id = f"{edge}_{i}"
                    length = float(self.traci.lane.getLength(lane_id))
                    totals[direction].append(length)
        except Exception:
            pass
        return {
            d: (sum(vals) / len(vals) if vals else 100.0)
            for d, vals in totals.items()
        }

    def _lane_static_data(
        self, incoming_edges: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Pre-compute per-lane direction and length for every lane on the
        given incoming edges.

        This MUST be called on the main thread during initialisation.
        The resulting dict is stored in junction_runtime["lane_data"] and
        used read-only from the background decision thread.  Caching here
        eliminates the need for any TraCI calls during decision cycles.

        TraCI call — must be called on the main thread.
        """
        result: Dict[str, Dict[str, Any]] = {}
        try:
            for edge in incoming_edges:
                num_lanes = self.traci.edge.getLaneNumber(edge)
                for i in range(num_lanes):
                    lane_id = f"{edge}_{i}"
                    result[lane_id] = {
                        "direction": _direction_from_angle(
                            _lane_heading(self.traci, lane_id)
                        ),
                        "length": float(self.traci.lane.getLength(lane_id)),
                    }
        except Exception:
            pass
        return result

    def _build_phase_map(self, incoming_edges: List[str]) -> Dict[str, str]:
        """Map each incoming edge to its phase ("NS" or "EW").

        TraCI call — must be called on the main thread.
        """
        result: Dict[str, str] = {}
        for edge in incoming_edges:
            direction = self._edge_direction(edge)
            result[edge] = "NS" if direction in {"N", "S"} else "EW"
        return result

    def _build_signal_groups(self, tls_id: str) -> Dict[str, List[int]]:
        """Map compass direction → list of controlled-link indices for a TLS.

        TraCI call — must be called on the main thread.
        """
        groups: Dict[str, List[int]] = {d: [] for d in config.DIRECTIONS}
        try:
            links = self.traci.trafficlight.getControlledLinks(tls_id)
            for index, link_group in enumerate(links):
                if not link_group:
                    continue
                incoming_lane = (
                    link_group[0][0] if link_group[0] else None
                )
                if not incoming_lane:
                    continue
                direction = _direction_from_angle(
                    _lane_heading(self.traci, incoming_lane)
                )
                groups[direction].append(index)
        except Exception as exc:
            logger.warning(
                "Could not build signal groups for TLS %s: %s", tls_id, exc
            )
        return {d: idxs for d, idxs in groups.items() if idxs}

    # ------------------------------------------------------------------
    # Per-step data helpers — read from DataCollectionLayer snapshots
    # These helpers are called from the BACKGROUND decision thread and
    # must NOT make any live TraCI calls.
    # ------------------------------------------------------------------

    def _junction_vehicle_records(
        self, junction_id: str
    ) -> List[Dict[str, Any]]:
        """Build the vehicle-record list the metric calculator expects.

        Reads only from data_collection (snapshot already committed by
        main thread).  No TraCI calls.  Safe to call from decision thread.
        """
        runtime = self.junction_runtime[junction_id]
        lane_data = runtime["lane_data"]
        records = []
        for raw in self.data_collection.get_all_vehicles():
            edge = str(raw.get("edge_id", ""))
            if edge not in runtime["incoming_edges"]:
                continue

            # Use the angle field from the raw vehicle dict for direction.
            # _direction_from_angle is a pure function — no TraCI call.
            direction = _direction_from_angle(
                float(raw.get("angle", 0.0) or 0.0)
            )

            lane_id = raw.get("lane_id")
            cached = lane_data.get(lane_id, {})
            # Prefer the cached direction (derived from lane geometry) over
            # the vehicle heading, which can be noisy mid-intersection.
            if cached.get("direction"):
                direction = cached["direction"]

            records.append({
                # Fields expected by TrafficMetricCalculator.calculate()
                "id": raw.get("vehicle_id"),
                "type": raw.get("type_id", "default"),
                "direction": direction,
                "speed": float(raw.get("speed", 0.0) or 0.0),
                "waiting_time": float(raw.get("waiting_time", 0.0) or 0.0),
                "lane_id": lane_id,
                # Fields expected by EmergencyVehicleDetection.update()
                "vehicle_id": raw.get("vehicle_id"),
                "vehicle_type": raw.get("type_id", "default"),
                "vehicle_class": raw.get("vehicle_class", ""),
                "current_edge": edge,
                "current_lane": lane_id,
                "controlled_tls": runtime["tls_id"],
                "junction_id": junction_id,
                "lane_length": cached.get("length", 100.0),
                "lane_position": float(raw.get("lane_position", 0.0) or 0.0),
                "length": float(raw.get("length", 5.0) or 5.0),
                "route": raw.get("route", []) or [],
                "route_index": raw.get("route_index", 0),
                "position": raw.get("position_xy"),
                "signals": raw.get("signals", 0),
            })
        return records

    def _blockage_input(
        self, junction_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """Build the lane-data dict BlockageDetection.update() expects.

        Reads only from data_collection.  No TraCI calls.
        Safe to call from decision thread.
        """
        runtime = self.junction_runtime[junction_id]
        result: Dict[str, Dict[str, Any]] = {}
        for lane in self.data_collection.get_all_lanes():
            edge = str(lane.get("edge_id", ""))
            if edge not in runtime["incoming_edges"]:
                continue
            lane_id = lane.get("lane_id")
            if not lane_id:
                continue
            ids: List[str] = list(lane.get("vehicle_ids", []) or [])
            lengths: Dict[str, float] = {}
            waits: Dict[str, float] = {}
            for vid in ids:
                v = self.data_collection.get_vehicle(vid) or {}
                lengths[vid] = float(v.get("length", 5.0) or 5.0)
                waits[vid] = float(v.get("waiting_time", 0.0) or 0.0)
            result[lane_id] = {
                "junction_id": junction_id,
                "lane_length": float(lane.get("length", 100.0) or 100.0),
                "max_speed": float(lane.get("max_speed", 0.0) or 0.0),
                "mean_speed": float(lane.get("mean_speed", 0.0) or 0.0),
                "halting_number": int(
                    lane.get("halting_vehicle_count", 0) or 0
                ),
                "vehicle_ids": ids,
                "vehicle_lengths": lengths,
                "vehicle_waiting_times": waits,
            }
        return result

    def _isolated_directions(self, junction_id: str) -> Set[str]:
        """Return the set of approach directions currently isolated.

        Uses the static lane_data cache (no TraCI call).
        Safe to call from decision thread.
        """
        lane_data = self.junction_runtime[junction_id]["lane_data"]
        isolated_lanes = (
            self.isolation_managers[junction_id]
            .get_output()
            .get("isolated_roads", [])
        )
        result: Set[str] = set()
        for lane_id in isolated_lanes:
            direction = lane_data.get(lane_id, {}).get("direction")
            if direction:
                result.add(direction)
        return result

    # ------------------------------------------------------------------
    # Decision cycle — runs in background thread; no live TraCI calls
    # ------------------------------------------------------------------

    def _process_junction(
        self, junction_id: str, sim_time: float
    ) -> Dict[str, Any]:
        """Run one full decision cycle for a single junction.

        Called from the background decision thread.  Must NOT make any
        live TraCI calls — reads only from data_collection snapshots and
        the static junction_runtime cache.
        """
        # --- Step 1: collect vehicle records from snapshot ---------------
        vehicles = self._junction_vehicle_records(junction_id)

        # --- Step 2: compute traffic metrics ----------------------------
        metrics = self.metric_calculators[junction_id].calculate(vehicles)

        # --- Step 3: blockage detection ---------------------------------
        blockage = self.blockage_detectors[junction_id].update(
            self._blockage_input(junction_id), sim_time
        )

        # --- Step 4: road isolation -------------------------------------
        lane_data = self.junction_runtime[junction_id]["lane_data"]
        for lane_id, report in blockage.items():
            try:
                self.isolation_managers[junction_id].process_report(report)
            except Exception as exc:
                # RoadIsolationError for malformed reports must be logged
                # clearly, not swallowed.
                logger.error(
                    "[%s] Road isolation error for lane %s: %s",
                    junction_id, lane_id, exc,
                )

            # Attach the blockage detector's queue_storage_ratio to the
            # traffic metrics so the QUBO builder can use it for congestion
            # mode selection.
            direction = lane_data.get(lane_id, {}).get("direction", "N")
            current = metrics.setdefault(direction, {})
            current["queue_storage_ratio"] = max(
                float(current.get("queue_storage_ratio", 0.0) or 0.0),
                float(report.get("queue_storage_ratio", 0.0) or 0.0),
            )

        # Ensure every direction entry exists with at least queue_storage_ratio.
        for direction in config.DIRECTIONS:
            metrics.setdefault(direction, {})
            metrics[direction].setdefault("queue_storage_ratio", 0.0)

        # --- Step 5: emergency detection --------------------------------
        # EmergencyVehicleDetection.update() expects:
        #   {"simulation_time": float, "vehicles": {vehicle_id: vehicle_dict}}
        emergency_snapshot = {
            "simulation_time": sim_time,
            "vehicles": {
                str(v["vehicle_id"]): v
                for v in vehicles
                if v.get("vehicle_id")
            },
        }
        emergency = self.emergency_detectors[junction_id].update(
            emergency_snapshot
        )

        # --- Step 6: predictor ------------------------------------------
        # predictor.predict(current_metrics,
        #                   upstream_junction_states=None,
        #                   corridor_graph=None)
        # Passing None for upstream/corridor is correct for a single-junction
        # run; extend here if corridor-graph data becomes available.
        predicted = self.predictors[junction_id].predict(metrics)

        # --- Step 7: isolated approaches --------------------------------
        isolated = self._isolated_directions(junction_id)

        # --- Step 8: decision engine ------------------------------------
        # The decision engine owns the full priority hierarchy:
        #   Emergency → Starvation → High-density → QUBO optimisation
        # It internally calls qubo_builder.build() and solver_manager.solve()
        # with exactly one solver per cycle.
        decision = self.decision_engines[junction_id].decide(
            current_metrics=metrics,
            predicted_metrics=predicted,
            emergency_result=emergency,
            isolated_approaches=isolated,
            junction_id=junction_id,
            solver_name=self.solver_name,
        )

        # --- Publish decision (thread-safe write) -----------------------
        with self._decision_lock:
            self.latest_decisions[junction_id] = decision

        # --- Log result -------------------------------------------------
        result = {
            "simulation_time": sim_time,
            "junction_id": junction_id,
            "selected_phase": decision.phase,
            "green_time": decision.green_time,
            "reason": decision.reason,
            "emergency": decision.emergency,
            "starvation_override": decision.starvation_override,
            "isolated_approaches": sorted(decision.isolated_approaches),
            "solver_result": decision.solver_result,
            "metrics": metrics,
            "predicted_metrics": predicted,
        }
        self._append_result(result)
        logger.info(
            "[%s] t=%.1f phase=%s green=%.1fs reason=%s solver=%s",
            junction_id,
            sim_time,
            decision.phase,
            decision.green_time,
            decision.reason,
            (decision.solver_result or {}).get("solver", "—"),
        )
        return result

    def _append_result(self, result: Dict[str, Any]) -> None:
        with self.results_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, default=str) + "\n")

    def _run_decision_cycle(self, sim_time: float) -> None:
        """Entry point for the background decision thread.

        Runs _process_junction() for every configured junction in sequence,
        then clears _decisions_in_flight.  No live TraCI calls are made.
        """
        for junction_id in list(self.junction_runtime.keys()):
            try:
                self._process_junction(junction_id, sim_time)
            except Exception:
                logger.exception(
                    "Decision cycle failed for junction %s at t=%.1f",
                    junction_id, sim_time,
                )
        self._decisions_in_flight = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start SUMO and initialise all runtime modules.

        All TraCI calls in this method and in _initialize_runtime_modules()
        run on the main thread, before the simulation loop begins.
        """
        if not os.environ.get("SUMO_HOME"):
            raise MainControllerError(
                "SUMO_HOME is not set. "
                "Set it to your SUMO installation directory before running."
            )
        self._load_modules()
        self.traci_api.start_simulation()
        # Import the live `traci` module AFTER start_simulation() so that
        # any usage of traci.* by helper methods below actually talks to
        # the running SUMO instance.
        self.traci = __import__("traci")
        self._initialize_runtime_modules()
        logger.info(
            "Controller started. Solver: %s, Decision interval: %.1fs",
            self.solver_name,
            self.decision_interval,
        )

    def run(self) -> None:
        """Main simulation loop.

        Threading contract:
          * Main thread calls simulationStep(), collect_step_data(),
            data_collection.update(), and signal_controller.apply().
          * Decision cycles run in a background thread so SUMO keeps
            receiving simulationStep() calls while a slow solver (QAOA)
            is computing.
          * The background thread only reads from the already-committed
            data_collection snapshot — no live TraCI calls.
        """
        self.start()

        # Determine the simulation end time from the SUMO configuration.
        # We keep the loop running until getMinExpectedNumber() == 0 AND
        # sim_time >= end_time, to avoid premature exit at t=0 before any
        # vehicles have departed.
        sumo_end_time = float(scenario_config.SIMULATION_END)

        try:
            # Always advance at least one step before checking is_running()
            # to avoid the "simulation ended at time 0.00" race where
            # getMinExpectedNumber() is briefly 0 before the first vehicle
            # has departed.
            sim_time = self.traci_api.step()
            snapshot = self._collect_fast_snapshot()
            self.data_collection.update(snapshot)

            while True:
                sim_time_f = float(sim_time) if sim_time is not None else 0.0

                # Stop when SUMO reports no more vehicles expected AND we
                # have passed the configured end time.
                if not self.traci_api.is_running() and sim_time_f >= sumo_end_time:
                    break

                # ---- Automated incident demonstration (t=150s to 450s) --
                self._handle_incident_injection(sim_time_f)

                # ---- Decision cycle (background thread) ----------------
                # Only start a new cycle if the previous one finished AND
                # the decision interval has elapsed.
                if (
                    sim_time_f >= self.next_decision_time
                    and not self._decisions_in_flight
                ):
                    self._decisions_in_flight = True
                    t = threading.Thread(
                        target=self._run_decision_cycle,
                        args=(sim_time_f,),
                        daemon=True,
                    )
                    t.start()
                    self._decision_threads.append(t)
                    self.decision_count += 1
                    self.next_decision_time = sim_time_f + self.decision_interval

                # ---- Signal controller tick (main thread, every step) --
                # SignalController must be ticked every simulation step so
                # its GREEN/YELLOW/ALL_RED state machine advances at SUMO's
                # step rate.  It calls traci.trafficlight.setRedYellowGreenState()
                # — this is a live TraCI call and MUST run on the main thread.
                with self._decision_lock:
                    current_decisions = dict(self.latest_decisions)

                for junction_id, controller in self.signal_controllers.items():
                    decision = current_decisions.get(junction_id)
                    if decision is None:
                        continue
                    # Force a phase switch immediately for emergency/starvation
                    # overrides even if the committed green time hasn't expired.
                    force_switch = (
                        decision.emergency or decision.starvation_override
                    ) and (
                        controller.current_phase != decision.phase
                        and controller.pending_phase != decision.phase
                    )
                    controller.apply(
                        decision.phase,
                        decision.green_time,
                        isolated_approaches=decision.isolated_approaches,
                        force=force_switch,
                    )

                # ---- Advance SUMO by one step --------------------------
                sim_time = self.traci_api.step()
                snapshot = self._collect_fast_snapshot()
                self.data_collection.update(snapshot)

        finally:
            # Wait for the in-flight decision thread before closing, so we
            # don't drop a partial result.
            for t in self._decision_threads:
                t.join(timeout=30)
            self.shutdown()

    def _handle_incident_injection(self, sim_time: float) -> None:
        """Inject a simulated accident/blockage on J3 cross-street at t=150s, clear at t=450s."""
        if 150.0 <= sim_time < 450.0:
            if not getattr(self, "_incident_active", False):
                self._incident_active = True
                logger.warning(
                    ">>> [INCIDENT DEMO] Simulating accident/blockage on J3 cross-street (1290479888#2) at t=%.1fs",
                    sim_time,
                )
                try:
                    vids = self.traci.edge.getLastStepVehicleIDs("1290479888#2")
                    if vids:
                        target_vid = vids[0]
                        self._stopped_vid = target_vid
                        self.traci.vehicle.setSpeed(target_vid, 0.0)
                        self.traci.vehicle.setColor(target_vid, (255, 0, 255, 255))
                        logger.warning(">>> [INCIDENT DEMO] Vehicle %s stopped as accident trigger", target_vid)
                    else:
                        self.traci.lane.setMaxSpeed("1290479888#2_0", 0.001)
                except Exception as exc:
                    logger.debug("Incident injection notice: %s", exc)
            elif hasattr(self, "_stopped_vid"):
                try:
                    self.traci.vehicle.setSpeed(self._stopped_vid, 0.0)
                except Exception:
                    pass
        elif sim_time >= 450.0 and getattr(self, "_incident_active", False):
            self._incident_active = False
            logger.info(">>> [INCIDENT DEMO] Incident cleared at t=%.1fs. Road re-opened.", sim_time)
            try:
                self.traci.lane.setMaxSpeed("1290479888#2_0", 13.89)
                if hasattr(self, "_stopped_vid"):
                    self.traci.vehicle.setSpeed(self._stopped_vid, -1.0)
            except Exception as exc:
                logger.debug("Incident clear notice: %s", exc)

    def shutdown(self) -> None:
        """Close the TraCI connection and log final statistics."""
        if self.traci_api is not None:
            try:
                self.traci_api.close_simulation()
            except Exception:
                logger.exception("TraCI shutdown failed")
        logger.info(
            "Simulation finished. Total decision cycles: %d", self.decision_count
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bengaluru Adaptive Traffic Controller"
    )
    parser.add_argument(
        "--sumo-config",
        default=scenario_config.SUMO_CONFIG,
        help="Path to the SUMO .sumocfg file.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch sumo-gui instead of headless sumo.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Decision interval in seconds (default: 30).",
    )
    parser.add_argument(
        "--solver",
        choices=["simulated_annealing", "qaoa", "adaptive"],
        default="adaptive",
        help=(
            "Solver to use. "
            "'adaptive' selects QAOA for high traffic "
            "(vehicle_count > config.HIGH_TRAFFIC_THRESHOLD) and "
            "Simulated Annealing otherwise. "
            "Exactly one solver executes per decision cycle."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional SUMO random seed.",
    )
    parser.add_argument(
        "--output-dir",
        default="simulation/results",
        help="Directory for decision_results.jsonl output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        MainController(
            sumo_config=args.sumo_config,
            use_gui=args.gui,
            decision_interval=args.interval,
            solver=args.solver,
            output_dir=args.output_dir,
            seed=args.seed,
        ).run()
    except Exception as exc:
        logger.exception("Controller failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())