# Read vehicle data from the Data Collection Layer.
# Identify emergency vehicles.
# Determine the required priority phase.
# Calculate how long the emergency phase should remain active.
# Notify the signal controller to override the normal adaptive logic.
# Return control to the normal controller once the emergency vehicle has cleared the intersection.
"""
emergency_vehicle_detection.py
================================================================================
Emergency Vehicle Detection & Prioritization Module
--------------------------------------------------------------------------------
Position in the pipeline:

    SUMO -> TraCI API -> traci_interface.py -> data_collection.py
         -> emergency_vehicle_detection.py (THIS MODULE) -> decision_engine.py
         -> signal_controller.py

This module is responsible ONLY for:
    - Detecting emergency vehicles from data supplied by the Data Collection
      Layer (data_collection.py).
    - Computing the required signal phase and an adaptive green duration for
      every detected emergency vehicle.
    - Managing and prioritizing multiple simultaneous emergency requests.
    - Broadcasting/receiving advance emergency information between adjacent
      junctions.
    - Activating an emergency override only after LOCAL confirmation.
    - Producing a single, well-defined "Emergency Decision" for the Decision
      Engine to consume.

This module MUST NEVER:
    - Talk to TraCI directly.
    - Control traffic lights directly.
    - Perform general traffic optimisation (that is decision_engine.py's job).

All traffic-state input arrives as plain Python dictionaries handed to this
module by data_collection.py. All output leaves this module as plain Python
dictionaries / dataclasses consumed by decision_engine.py (or, for
inter-junction communication, by whatever orchestrator wires multiple
EmergencyVehicleDetection instances together).
================================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================
# All tunable parameters live here so behaviour can be adjusted without
# touching the detection / prioritization / timing logic below.
# ============================================================================

class EmergencyConfig:
    """Centralized, overridable configuration for emergency vehicle handling.

    NOTE: every tunable below is assigned per-instance in __init__ rather
    than as a class-level attribute. dict/set class attributes are shared
    by ALL instances of a class unless each instance gets its own copy;
    since one EmergencyConfig() is created per junction (see
    EmergencyVehicleDetection.__init__), a class-level dict here would mean
    an in-place update to one junction's NEIGHBOR_JUNCTIONS or
    JUNCTION_PHASE_MAP could silently leak into every other junction's
    configuration. Per-instance copies make each junction's config fully
    independent.
    """

    def __init__(self) -> None:
        # ---- Vehicle identification -----------------------------------
        # A vehicle is considered "emergency" if EITHER its SUMO vehicle class
        # OR its vehicle type matches one of these configurable sets.
        self.EMERGENCY_VEHICLE_CLASSES: Set[str] = {"emergency"}
        self.EMERGENCY_VEHICLE_TYPES: Set[str] = {"ambulance", "fire_truck", "police"}

        # Used only as the 4th (lowest-weight) prioritization tiebreaker.
        # Higher number = higher priority.
        self.VEHICLE_TYPE_PRIORITY: Dict[str, int] = {
            "fire_truck": 3,
            "ambulance": 2,
            "police": 1,
        }

        # ---- Green duration calculation --------------------------------
        self.SAFETY_BUFFER_TIME: float = 3.0            # seconds, added safety margin
        self.INTERSECTION_CROSSING_DISTANCE: float = 20.0   # meters, stop line -> fully clear
        self.MIN_GREEN_DURATION: float = 5.0            # seconds, hard floor
        self.MAX_GREEN_DURATION: float = 60.0           # seconds, hard ceiling
        self.FALLBACK_SPEED: float = 8.33               # m/s (~30 km/h), used if reported speed ~ 0
        self.QUEUE_DELAY_PER_VEHICLE: float = 2.0       # seconds added per queued vehicle ahead
        self.STOPPED_SPEED_EPSILON: float = 0.5         # m/s, below this we treat vehicle as stopped
        self.DEFAULT_STOP_LINE_DISTANCE: float = 50.0   # meters, fallback when unknown

        # ---- Standby / broadcast lifecycle -------------------------------
        self.STANDBY_EXPIRY_TIME: float = 60.0          # seconds; drop stale un-confirmed broadcasts

        # ---- Prioritization -----------------------------------------------
        # Order of tiebreakers, in priority order (first = most significant).
        # Values must correspond to fields understood by _priority_sort_key().
        self.PRIORITY_ORDER: Tuple[str, ...] = ("eta", "distance", "waiting_time", "vehicle_type")

        # ---- Network topology & signal geometry (must be supplied per-deployment) --
        # junction_id -> list of adjacent/connected junction ids to broadcast to.
        self.NEIGHBOR_JUNCTIONS: Dict[str, List[str]] = {}

        # junction_id -> { incoming_edge_id: phase_id_or_index }
        # Defines which signal phase serves each incoming approach at a junction.
        self.JUNCTION_PHASE_MAP: Dict[str, Dict[str, Any]] = {}


class EmergencyStatus(str, Enum):
    """Lifecycle status of an emergency request."""
    DETECTED = "DETECTED"          # seen locally, request built, not yet prioritized
    ACTIVE = "ACTIVE"               # locally confirmed, currently the (or a) live request
    STANDBY = "STANDBY"             # received from a neighbour, not yet locally confirmed
    CLEARING = "CLEARING"           # vehicle has entered the intersection, being watched out
    CLEARED = "CLEARED"             # vehicle has fully cleared / left, request retired


# ============================================================================
# SECTION 2: DATA STRUCTURES
# ============================================================================

@dataclass
class EmergencyRequest:
    """Full internal representation of one emergency vehicle's request at THIS junction."""
    vehicle_id: str
    vehicle_type: str
    junction_id: str
    lane_id: str
    edge_id: str
    position: Tuple[float, float]
    speed: float
    waiting_time: float
    distance_to_stop_line: float
    eta: float
    required_phase: Optional[Any]
    green_duration: float
    priority: Optional[int]              # 1 = highest; assigned during prioritization
    status: EmergencyStatus
    last_updated: float                   # simulation time of last refresh

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class BroadcastMessage:
    """
    Advance-information-only message sent to adjacent junctions.

    Deliberately EXCLUDES: final green duration, final phase decision, and any
    signal-controller commands -- those are local-only, per STEP 9 spec.
    """
    vehicle_id: str
    vehicle_type: str
    origin_junction: str
    position: Tuple[float, float]
    speed: float
    eta: float
    priority: Optional[int]
    status: str
    timestamp: float
    target_junction: str = ""   # filled in when addressed to a specific neighbour

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# SECTION 3: MAIN CLASS
# ============================================================================

class EmergencyVehicleDetection:
    """
    Emergency vehicle detection, prioritization and inter-junction coordination
    for a single junction. One instance should be created per controlled
    junction; a thin orchestrator is expected to route BroadcastMessage objects
    between instances that represent adjacent junctions (this module never
    talks across junctions on its own -- it only produces/consumes messages).
    """

    def __init__(
        self,
        junction_id: str,
        config: Optional[EmergencyConfig] = None,
        neighbor_junctions: Optional[Sequence[str]] = None,
        phase_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.junction_id = junction_id
        self.config = config or EmergencyConfig()

        # Resolve topology / geometry, allowing explicit overrides per instance.
        self.neighbor_junctions: List[str] = list(
            neighbor_junctions
            if neighbor_junctions is not None
            else self.config.NEIGHBOR_JUNCTIONS.get(junction_id, [])
        )
        self.phase_map: Dict[str, Any] = (
            phase_map
            if phase_map is not None
            else self.config.JUNCTION_PHASE_MAP.get(junction_id, {})
        )

        # Local, confirmed requests (detected by THIS junction's own sensors).
        self.active_requests: Dict[str, EmergencyRequest] = {}

        # Requests received via broadcast from neighbours, not yet locally confirmed.
        self.standby_requests: Dict[str, EmergencyRequest] = {}

        # Outboxes drained by the orchestrator after each update().
        self._outgoing_broadcasts: List[BroadcastMessage] = []
        self._outgoing_clearances: List[Dict[str, Any]] = []

        self._last_sim_time: float = 0.0

    # ------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------
    def update(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main per-simulation-step entry point.

        Args:
            snapshot: dictionary produced by data_collection.py, expected shape:
                {
                    "simulation_time": float,
                    "vehicles": {
                        vehicle_id: {
                            "vehicle_type": str,
                            "vehicle_class": str,
                            "current_edge": str,
                            "current_lane": str,
                            "current_road": str,
                            "junction_id": str,
                            "controlled_tls": str,
                            "position": (x, y),
                            "speed": float,
                            "waiting_time": float,
                            "route": List[str],
                            "route_index": int,
                            "lane_length": float,     # optional, for stop-line distance
                            "lane_position": float,   # optional, for stop-line distance
                            "queue_ahead": int,        # optional, vehicles ahead in queue
                        },
                        ...
                    }
                }

        Returns:
            The structured OUTPUT dictionary described in the module spec.
        """
        sim_time = float(snapshot.get("simulation_time", self._last_sim_time))
        self._last_sim_time = sim_time
        vehicles: Dict[str, Dict[str, Any]] = snapshot.get("vehicles", {})

        # STEP 2/3: detect emergency vehicles relevant to this junction and
        # build/refresh their EmergencyRequest.
        self._detect_and_update_local_requests(vehicles, sim_time)

        # STEP 9/14: check which previously-active local vehicles have cleared,
        # remove them and queue clearance broadcasts.
        self._detect_cleared_vehicles(vehicles, sim_time)

        # STEP 10: try to promote standby (neighbour-broadcast) requests into
        # active local requests if we can now see them locally ourselves.
        self._reconcile_standby_with_local(vehicles, sim_time)

        # Drop stale standby requests nobody ever confirmed.
        self._expire_stale_standby(sim_time)

        # STEP 7: prioritize all currently active local requests.
        priority_order = self._prioritize_active_requests()

        # STEP 9: queue outbound broadcasts for currently active local requests.
        self._queue_broadcasts(sim_time)

        return self._build_output(priority_order)

    # ------------------------------------------------------------------
    # STEP 2/3/4/5/6: DETECTION, PHASE, DURATION, REQUEST CONSTRUCTION
    # ------------------------------------------------------------------
    def _detect_and_update_local_requests(
        self, vehicles: Dict[str, Dict[str, Any]], sim_time: float
    ) -> None:
        """Scan the snapshot for emergency vehicles relevant to this junction."""
        for vehicle_id, info in vehicles.items():
            if not self._is_emergency_vehicle(info):
                continue
            if not self._is_relevant_to_this_junction(info):
                continue
            if not self._is_still_approaching(info):
                # Already past this junction's control zone; clearance handled
                # separately in _detect_cleared_vehicles().
                continue

            required_phase = self._determine_required_phase(info)
            distance = self._get_distance_to_stop_line(info)
            speed = self._safe_speed(info.get("speed", 0.0))
            eta = self._estimate_eta(distance, speed)
            queue_ahead = int(info.get("queue_ahead", 0) or 0)
            green_duration = self._calculate_green_duration(distance, speed, queue_ahead)

            self.active_requests[vehicle_id] = EmergencyRequest(
                vehicle_id=vehicle_id,
                vehicle_type=str(info.get("vehicle_type", "")),
                junction_id=self.junction_id,
                lane_id=str(info.get("current_lane", "")),
                edge_id=str(info.get("current_edge", "")),
                position=tuple(info.get("position", (0.0, 0.0))),
                speed=speed,
                waiting_time=float(info.get("waiting_time", 0.0) or 0.0),
                distance_to_stop_line=distance,
                eta=eta,
                required_phase=required_phase,
                green_duration=green_duration,
                priority=None,   # assigned in _prioritize_active_requests
                status=EmergencyStatus.DETECTED,
                last_updated=sim_time,
            )

    def _is_emergency_vehicle(self, info: Dict[str, Any]) -> bool:
        """STEP 2: configurable rule-based emergency vehicle detection."""
        vehicle_class = str(info.get("vehicle_class", "")).lower()
        vehicle_type = str(info.get("vehicle_type", "")).lower()
        return (
            vehicle_class in self.config.EMERGENCY_VEHICLE_CLASSES
            or vehicle_type in self.config.EMERGENCY_VEHICLE_TYPES
        )

    def _is_relevant_to_this_junction(self, info: Dict[str, Any]) -> bool:
        """
        A vehicle is relevant to this junction if the Data Collection Layer
        reports it as approaching this junction's controlled traffic light,
        OR its current edge appears in this junction's phase map (incoming
        approach list) as a fallback when junction_id is not explicitly set.
        """
        reported_junction = info.get("junction_id") or info.get("controlled_tls")
        if reported_junction:
            return str(reported_junction) == self.junction_id
        return str(info.get("current_edge", "")) in self.phase_map

    def _is_still_approaching(self, info: Dict[str, Any]) -> bool:
        """
        True while the vehicle's current edge is a known incoming approach for
        this junction (i.e. it has not yet passed through / been assigned to an
        outgoing edge).
        """
        edge = str(info.get("current_edge", ""))
        if not self.phase_map:
            # No geometry configured -- assume still approaching; caller should
            # configure JUNCTION_PHASE_MAP for accurate clearance detection.
            return True
        return edge in self.phase_map

    def _determine_required_phase(self, info: Dict[str, Any]) -> Optional[Any]:
        """STEP 4: look up the signal phase serving this vehicle's approach."""
        edge = str(info.get("current_edge", ""))
        return self.phase_map.get(edge)

    def _get_distance_to_stop_line(self, info: Dict[str, Any]) -> float:
        """
        Prefer an explicit distance if the Data Collection Layer supplies one;
        otherwise derive it from lane length / lane position; otherwise fall
        back to a configured default.
        """
        if "distance_to_stop_line" in info:
            try:
                return max(0.0, float(info["distance_to_stop_line"]))
            except (TypeError, ValueError):
                pass
        lane_length = info.get("lane_length")
        lane_position = info.get("lane_position")
        if lane_length is not None and lane_position is not None:
            try:
                return max(0.0, float(lane_length) - float(lane_position))
            except (TypeError, ValueError):
                pass
        return self.config.DEFAULT_STOP_LINE_DISTANCE

    def _safe_speed(self, speed: float) -> float:
        """Avoid division-by-zero: treat near-stationary vehicles as moving at
        the configured fallback speed for ETA/duration purposes."""
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            speed = 0.0
        if speed < self.config.STOPPED_SPEED_EPSILON:
            return self.config.FALLBACK_SPEED
        return speed

    def _estimate_eta(self, distance_to_stop_line: float, speed: float) -> float:
        """STEP 3: Estimated Arrival Time at the stop line, in seconds."""
        return distance_to_stop_line / speed

    def _calculate_green_duration(
        self, distance_to_stop_line: float, speed: float, queue_ahead: int
    ) -> float:
        """
        STEP 5: dynamically compute the green duration required for the
        emergency vehicle to reach the stop line, enter the intersection and
        completely clear it -- NEVER a fixed value.
        """
        time_to_stop_line = distance_to_stop_line / speed
        time_to_clear_intersection = (
            self.config.INTERSECTION_CROSSING_DISTANCE / speed
        )
        queue_delay = queue_ahead * self.config.QUEUE_DELAY_PER_VEHICLE

        duration = (
            time_to_stop_line
            + time_to_clear_intersection
            + queue_delay
            + self.config.SAFETY_BUFFER_TIME
        )

        # Clamp to sane operational bounds.
        duration = max(self.config.MIN_GREEN_DURATION, duration)
        duration = min(self.config.MAX_GREEN_DURATION, duration)
        return duration

    # ------------------------------------------------------------------
    # STEP 9 / STEP 14: CLEARANCE DETECTION
    # ------------------------------------------------------------------
    def _detect_cleared_vehicles(
        self, vehicles: Dict[str, Dict[str, Any]], sim_time: float
    ) -> None:
        """
        A locally-active vehicle is considered "cleared" when:
          (a) it no longer appears in the snapshot at all (left the
              simulation / completed its route), OR
          (b) it is still present but its current edge is no longer one of
              this junction's known incoming approaches (i.e. it moved on).
        """
        cleared_ids: List[str] = []
        for vehicle_id, request in self.active_requests.items():
            info = vehicles.get(vehicle_id)
            if info is None:
                cleared_ids.append(vehicle_id)
                continue
            if not self._is_still_approaching(info):
                cleared_ids.append(vehicle_id)

        for vehicle_id in cleared_ids:
            request = self.active_requests.pop(vehicle_id)
            request.status = EmergencyStatus.CLEARED
            self._outgoing_clearances.append(
                {
                    "vehicle_id": vehicle_id,
                    "junction_id": self.junction_id,
                    "timestamp": sim_time,
                    "type": "CLEARANCE",
                }
            )
            self.standby_requests.pop(vehicle_id, None)

    # ------------------------------------------------------------------
    # STEP 10 / STEP 11: STANDBY HANDLING + LOCAL CONFIRMATION
    # ------------------------------------------------------------------
    def receive_broadcast(self, message: BroadcastMessage) -> None:
        """
        STEP 10: Receive an advance EmergencyRequest broadcast from a
        neighbouring junction's EmergencyVehicleDetection instance. The
        request is stored in STANDBY only -- emergency override is NEVER
        activated from a broadcast alone.
        """
        if message.vehicle_id in self.active_requests:
            # Already locally confirmed; broadcast info is now redundant.
            return

        existing = self.standby_requests.get(message.vehicle_id)
        if existing is not None:
            # Update ETA / position with the freshest broadcast data.
            existing.eta = message.eta
            existing.position = message.position
            existing.speed = message.speed
            existing.last_updated = message.timestamp
            return

        self.standby_requests[message.vehicle_id] = EmergencyRequest(
            vehicle_id=message.vehicle_id,
            vehicle_type=message.vehicle_type,
            junction_id=self.junction_id,
            lane_id="",
            edge_id="",
            position=message.position,
            speed=message.speed,
            waiting_time=0.0,
            distance_to_stop_line=self.config.DEFAULT_STOP_LINE_DISTANCE,
            eta=message.eta,
            required_phase=None,
            green_duration=0.0,
            priority=None,
            status=EmergencyStatus.STANDBY,
            last_updated=message.timestamp,
        )

    def _reconcile_standby_with_local(
        self, vehicles: Dict[str, Dict[str, Any]], sim_time: float
    ) -> None:
        """
        STEP 11: Local confirmation. A standby request is only promoted to an
        active, locally-computed request once THIS junction's own Data
        Collection snapshot actually contains the vehicle. The previous
        junction's green duration is never reused -- everything (phase,
        distance, ETA, duration) is recalculated purely from local data.
        """
        confirmed_ids: List[str] = []
        for vehicle_id, standby in self.standby_requests.items():
            info = vehicles.get(vehicle_id)
            if info is None:
                continue
            if not self._is_emergency_vehicle(info):
                continue
            if not self._is_relevant_to_this_junction(info):
                continue
            if not self._is_still_approaching(info):
                continue

            # Recompute everything locally -- do not trust/reuse standby data.
            required_phase = self._determine_required_phase(info)
            distance = self._get_distance_to_stop_line(info)
            speed = self._safe_speed(info.get("speed", 0.0))
            eta = self._estimate_eta(distance, speed)
            queue_ahead = int(info.get("queue_ahead", 0) or 0)
            green_duration = self._calculate_green_duration(distance, speed, queue_ahead)

            self.active_requests[vehicle_id] = EmergencyRequest(
                vehicle_id=vehicle_id,
                vehicle_type=str(info.get("vehicle_type", standby.vehicle_type)),
                junction_id=self.junction_id,
                lane_id=str(info.get("current_lane", "")),
                edge_id=str(info.get("current_edge", "")),
                position=tuple(info.get("position", (0.0, 0.0))),
                speed=speed,
                waiting_time=float(info.get("waiting_time", 0.0) or 0.0),
                distance_to_stop_line=distance,
                eta=eta,
                required_phase=required_phase,
                green_duration=green_duration,
                priority=None,
                status=EmergencyStatus.DETECTED,
                last_updated=sim_time,
            )
            confirmed_ids.append(vehicle_id)

        for vehicle_id in confirmed_ids:
            del self.standby_requests[vehicle_id]

    def _expire_stale_standby(self, sim_time: float) -> None:
        """Drop standby requests that were never locally confirmed in time."""
        expired = [
            vid
            for vid, req in self.standby_requests.items()
            if sim_time - req.last_updated > self.config.STANDBY_EXPIRY_TIME
        ]
        for vid in expired:
            del self.standby_requests[vid]

    # ------------------------------------------------------------------
    # STEP 7: PRIORITIZATION
    # ------------------------------------------------------------------
    def _priority_sort_key(self, request: EmergencyRequest) -> Tuple[float, float, float, int]:
        """
        Builds the sort key according to config.PRIORITY_ORDER. Default order:
            1. Lowest ETA
            2. Shortest distance to stop line
            3. Highest waiting time
            4. Highest vehicle-type priority (Fire Truck > Ambulance > Police)
        """
        type_priority = self.config.VEHICLE_TYPE_PRIORITY.get(request.vehicle_type, 0)
        # Negate fields where "higher is better" so a plain ascending sort works
        # uniformly for every field.
        field_values = {
            "eta": request.eta,
            "distance": request.distance_to_stop_line,
            "waiting_time": -request.waiting_time,
            "vehicle_type": -type_priority,
        }
        return tuple(field_values[name] for name in self.config.PRIORITY_ORDER)  # type: ignore[return-value]

    def _prioritize_active_requests(self) -> List[str]:
        """
        STEP 7: rank all active local requests. Only the highest-priority
        request is forwarded to the Decision Engine as the "current" request;
        the rest remain active (visible in all_active_requests) until it is
        their turn.
        """
        ordered = sorted(self.active_requests.values(), key=self._priority_sort_key)
        for rank, request in enumerate(ordered, start=1):
            request.priority = rank
            request.status = (
                EmergencyStatus.ACTIVE if rank == 1 else EmergencyStatus.DETECTED
            )
        return [r.vehicle_id for r in ordered]

    # ------------------------------------------------------------------
    # STEP 9: BROADCAST TO ADJACENT JUNCTIONS
    # ------------------------------------------------------------------
    def _queue_broadcasts(self, sim_time: float) -> None:
        """
        Build advance-information-only BroadcastMessages for every active
        local request and queue one per configured neighbour junction. Final
        green duration / phase / controller commands are intentionally
        excluded (see BroadcastMessage docstring).
        """
        if not self.neighbor_junctions:
            return
        for request in self.active_requests.values():
            for neighbor_id in self.neighbor_junctions:
                self._outgoing_broadcasts.append(
                    BroadcastMessage(
                        vehicle_id=request.vehicle_id,
                        vehicle_type=request.vehicle_type,
                        origin_junction=self.junction_id,
                        position=request.position,
                        speed=request.speed,
                        eta=request.eta,
                        priority=request.priority,
                        status=request.status.value,
                        timestamp=sim_time,
                        target_junction=neighbor_id,
                    )
                )

    def get_pending_broadcasts(self) -> List[BroadcastMessage]:
        """Drain and return outbound broadcast messages for the orchestrator
        to deliver to the relevant neighbour EmergencyVehicleDetection
        instances (via their receive_broadcast() method)."""
        messages = self._outgoing_broadcasts
        self._outgoing_broadcasts = []
        return messages

    def get_pending_clearances(self) -> List[Dict[str, Any]]:
        """Drain and return clearance notifications to broadcast to neighbours."""
        clearances = self._outgoing_clearances
        self._outgoing_clearances = []
        return clearances

    # ------------------------------------------------------------------
    # STEP 12/13: EMERGENCY DECISION OUTPUT
    # ------------------------------------------------------------------
    def _build_output(self, priority_order: List[str]) -> Dict[str, Any]:
        """
        STEP 12/13: assemble the structured output consumed by the Decision
        Engine. Only the highest-priority, LOCALLY CONFIRMED request becomes
        the active emergency override; everything else is informational.
        """
        emergency_active = len(priority_order) > 0
        current_request: Optional[EmergencyRequest] = (
            self.active_requests[priority_order[0]] if emergency_active else None
        )

        return {
            "emergency_active": emergency_active,
            "current_active_request": (
                current_request.to_dict() if current_request else None
            ),
            "all_active_requests": {
                vid: req.to_dict() for vid, req in self.active_requests.items()
            },
            "priority_order": priority_order,
            "override_phase": current_request.required_phase if current_request else None,
            "green_duration": current_request.green_duration if current_request else 0.0,
        }

    # ------------------------------------------------------------------
    # Convenience / introspection helpers
    # ------------------------------------------------------------------
    def get_standby_requests(self) -> Dict[str, Dict[str, Any]]:
        """Expose current standby (unconfirmed neighbour) requests, e.g. for logging."""
        return {vid: req.to_dict() for vid, req in self.standby_requests.items()}


# ============================================================================
# SECTION 4: LIGHTWEIGHT DEMONSTRATION (no TraCI / SUMO involved)
# ============================================================================
if __name__ == "__main__":
    # This block only demonstrates the module's internal logic using synthetic
    # data shaped like what data_collection.py would supply. It performs no
    # simulation and no TraCI calls whatsoever.

    demo_config = EmergencyConfig()
    demo_config.JUNCTION_PHASE_MAP = {
        "J1": {"edge_in_north": "phase_NS", "edge_in_south": "phase_NS"},
    }
    demo_config.NEIGHBOR_JUNCTIONS = {"J1": ["J2"]}

    evd_j1 = EmergencyVehicleDetection(junction_id="J1", config=demo_config)

    snapshot_t0 = {
        "simulation_time": 100.0,
        "vehicles": {
            "amb_1": {
                "vehicle_type": "ambulance",
                "vehicle_class": "emergency",
                "current_edge": "edge_in_north",
                "current_lane": "edge_in_north_0",
                "current_road": "North Ave",
                "junction_id": "J1",
                "controlled_tls": "J1",
                "position": (10.0, 200.0),
                "speed": 12.0,
                "waiting_time": 0.0,
                "route": ["edge_in_north", "edge_out_south"],
                "route_index": 0,
                "lane_length": 150.0,
                "lane_position": 30.0,
                "queue_ahead": 2,
            }
        },
    }

    result = evd_j1.update(snapshot_t0)
    print("Emergency active:", result["emergency_active"])
    print("Override phase:", result["override_phase"])
    print("Green duration (s):", round(result["green_duration"], 2))

    # Simulate the vehicle clearing the junction on the next step.
    snapshot_t1 = {"simulation_time": 130.0, "vehicles": {}}
    result_cleared = evd_j1.update(snapshot_t1)
    print("Emergency active after clearance:", result_cleared["emergency_active"])
    print("Pending clearances:", evd_j1.get_pending_clearances())