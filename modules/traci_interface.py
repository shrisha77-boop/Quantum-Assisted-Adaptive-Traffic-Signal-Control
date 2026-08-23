# Start SUMO.

# ↓

# Connect to TraCI.

# ↓

# Advance the simulation by one step.

# ↓

# Read raw information from TraCI.

# ↓

# Return the collected information to
# Data Collection.

# ↓

# Close the simulation.
"""
traci_interface.py
===================

Pure TraCI API layer for the adaptive traffic signal control system.

This file's ONLY job is to talk to SUMO over TraCI and hand back RAW values
as plain Python dicts/lists. It performs NO calculations, aggregation,
prediction, or decision-making of any kind. Every value returned here is
either:
    (a) retrieved directly from a TraCI getter, or
    (b) explicitly marked as "not available via TraCI" when SUMO's API
        does not expose it directly (per the project requirement to never
        invent or approximate a value).


Responsibilities of this file (and ONLY these):
    1. Connect to the SUMO simulation.
    2. Start and close the TraCI connection.
    3. Advance the simulation by one step.
    4. Read raw information from the TraCI API.
    5. Return that raw information as clean Python dicts/lists.
    6. Handle TraCI exceptions safely (vehicle disappeared, lane/edge
       unavailable, simulation finished, connection dropped, etc.)

Requires: SUMO_HOME environment variable set, and the `traci` package that
ships with SUMO (bootstrapped onto sys.path below if not already importable).
"""

import os
import sys
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# ---------------------------------------------------------------------------
# Make sure traci is importable (standard SUMO_HOME bootstrap)
# ---------------------------------------------------------------------------
if "SUMO_HOME" in os.environ:
    tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)
else:
    raise EnvironmentError(
        "Please declare the SUMO_HOME environment variable "
        "(e.g. export SUMO_HOME='/usr/share/sumo')."
    )

import traci  # noqa: E402
from traci.exceptions import TraCIException, FatalTraCIError  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("traci_interface")

_UNAVAILABLE = "UNAVAILABLE_VIA_TRACI"  # explicit marker, never a guessed value


# ---------------------------------------------------------------------------
# Reference-only lookup tables.
# The Data Collection Layer is free to use
# or ignore these when interpreting the raw values.
# ---------------------------------------------------------------------------
_TRACI_DIRECTION_MAP = {
    "s": "straight",
    "l": "left",
    "r": "right",
    "L": "partial-left",
    "R": "partial-right",
    "t": "u-turn",
    "invalid": "invalid",
}

# Bit meanings for traci.vehicle.getSignals(vehID), as defined by SUMO's own
# tc.VEH_SIGNAL_* constants. Bit 11 (value 2048) is the emergency blue light.
VEHICLE_SIGNAL_BIT_REFERENCE = {
    0: "blinker_right",
    1: "blinker_left",
    2: "blinker_emergency",
    3: "brake_light",
    4: "front_light",
    5: "fog_light",
    6: "high_beam",
    7: "backdrive_light",
    8: "wiper",
    9: "door_open_left",
    10: "door_open_right",
    11: "emergency_blue_light",
    12: "emergency_red_light",
    13: "emergency_yellow_light",
}

_COMPASS_SECTORS = [
    "north", "north-east", "east", "south-east",
    "south", "south-west", "west", "north-west",
]


def angle_to_compass(angle_deg: float) -> str:
    """Convert a SUMO heading angle (0=N, clockwise) to a compass sector."""
    idx = int(((angle_deg + 22.5) % 360) / 45)
    return _COMPASS_SECTORS[idx]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TraCIConfig:
    sumocfg_path: str
    use_gui: bool = False
    sumo_binary_override: Optional[str] = None
    port: Optional[int] = None
    step_length: float = 2.0
    extra_sumo_args: List[str] = field(default_factory=list)

    # --- GUI behaviour -----------------------------------------------------
    # sumo-gui loads a scenario PAUSED by default and waits for the user to
    # click "Play" -- it does not auto-run just because a TraCI client is
    # driving it with simulationStep() calls, so without SUMO's own --start
    # flag the window opens, sits at time 0.0, and vehicles never move until
    # someone clicks Play by hand. auto_start_gui reproduces the behaviour
    # of running `sumo-gui ... --start` directly. gui_delay_ms mirrors
    # sumo-gui's own --delay (ms paused between rendered steps) purely so
    # the GUI is watchable instead of racing through frames as fast as the
    # Python-side per-step processing allows; it has no effect in headless
    # (use_gui=False) mode.
    auto_start_gui: bool = True
    gui_delay_ms: int = 0


    # --- Snapshot collection toggles -------------------------------------
    # collect_step_data() defaults to collecting every domain (unchanged
    # public behaviour / back-compat), but a full-city OSM network has tens
    # of thousands of edges/lanes/junctions, and re-querying all of them
    # over TraCI every single simulation step is expensive. These toggles
    # let a caller that doesn't consume a given domain (e.g. this project's
    # own main_controller.py never reads snapshot["edges"] or
    # snapshot["turning_movements"]) opt out of collecting it, without
    # changing collect_step_data()'s return shape (skipped domains are
    # still present as empty lists) or removing any of the underlying
    # get_all_*_data() methods, which remain fully available for direct use.
    collect_edges: bool = False
    collect_junctions: bool = False
    collect_traffic_lights: bool = True
    collect_turning_movements: bool = False


# ---------------------------------------------------------------------------
# Main interface class
# ---------------------------------------------------------------------------
class TraCIInterface:
    """
    Pure wrapper around the SUMO TraCI API. Every `get_*` method maps to one
    or more direct TraCI getters and returns raw values only.
    """

    def __init__(self, config: TraCIConfig):
        self.config = config
        self._connected = False
        self._lane_angle_cache: Dict[str, float] = {}  # perf: computed once per lane

    # ------------------------------------------------------------------
    # Generic safe-call helper -- centralises exception handling so every
    # get_* method doesn't repeat the same try/except boilerplate.
    # Returns `default` and logs a warning if the TraCI call fails for any
    # reason (entity removed/disappeared, domain unsupported, disconnected).
    # ------------------------------------------------------------------
    def _safe(self, func, *args, default=_UNAVAILABLE, **kwargs):
        try:
            return func(*args, **kwargs)
        except (TraCIException, FatalTraCIError, AttributeError, KeyError) as exc:
            logger.debug("TraCI call %s failed: %s", getattr(func, "__name__", func), exc)
            return default

    # ------------------------------------------------------------------
    # Lifecycle: start / step / running-check / close
    # ------------------------------------------------------------------
    def start_simulation(self) -> None:
        """Builds the SUMO command line and opens the TraCI connection."""
        binary = self.config.sumo_binary_override or (
            "sumo-gui" if self.config.use_gui else "sumo"
        )
        sumo_cmd = [
            binary,
            "-c", self.config.sumocfg_path,
            "--step-length", str(self.config.step_length),
            "--no-step-log", "true",
            "--duration-log.disable", "true",
        ]
        if self.config.use_gui and self.config.auto_start_gui:
            sumo_cmd += ["--start"]
            if self.config.gui_delay_ms > 0:
                sumo_cmd += ["--delay", str(self.config.gui_delay_ms)]

        sumo_cmd += self.config.extra_sumo_args

        logger.info("Starting SUMO: %s", " ".join(sumo_cmd))
        try:
            if self.config.port is not None:
                traci.start(sumo_cmd, port=self.config.port)
            else:
                traci.start(sumo_cmd)
            self._connected = True
            logger.info("Connected to TraCI successfully.")
        except FatalTraCIError as exc:
            logger.error("Failed to start/connect to TraCI: %s", exc)
            raise

    def step(self) -> float:
        """
        Advances the simulation by one step.
        Raw TraCI call: traci.simulationStep()
        Returns: the new simulation time (traci.simulation.getTime()).
        """
        try:
            traci.simulationStep()
        except FatalTraCIError as exc:
            logger.error("Simulation step failed (connection may be closed): %s", exc)
            raise
        return self._safe(traci.simulation.getTime, default=None)

    def is_running(self) -> bool:
        """
        Raw TraCI call: traci.simulation.getMinExpectedNumber()
        Returns True while SUMO still expects vehicles to be departing/active.
        Used by the caller to decide whether to keep looping -- not a decision
        made by this interface.
        """
        return self._safe(traci.simulation.getMinExpectedNumber, default=0) > 0

    def close_simulation(self) -> None:
        """Raw TraCI call: traci.close(). Closes the connection safely."""
        if self._connected:
            try:
                traci.close()
            except Exception as exc:
                logger.warning("Error while closing TraCI connection: %s", exc)
            finally:
                self._connected = False
                logger.info("TraCI connection closed.")

    # ==================================================================
    # 1. SIMULATION-LEVEL RAW DATA
    # ==================================================================
    def get_simulation_data(self) -> Dict[str, Any]:
        """
        Uses: traci.simulation.getTime, getDeltaT, getMinExpectedNumber,
              getLoadedIDList/Number, getDepartedIDList/Number,
              getArrivedIDList/Number, getStartingTeleportIDList/Number,
              getEndingTeleportIDList/Number, getCollidingVehiclesIDList/Number,
              getNetBoundary.

        Returns raw simulation-level counters/lists for the current step.

        Notes on fields the user asked for that TraCI does NOT expose directly:
          - "Simulation step" (an integer step counter) is not a separate raw
            getter. It is derivable downstream as time / delta_t, but this
            interface does not compute that -- only current_time and
            delta_time are returned.
          - "Current simulation state" as a single raw value does not exist
            in TraCI; use is_running() (based on getMinExpectedNumber) if a
            running/finished check is needed.
        """
        return {
            "current_time": self._safe(traci.simulation.getTime),
            "delta_time": self._safe(traci.simulation.getDeltaT),
            "min_expected_number": self._safe(traci.simulation.getMinExpectedNumber),
            "loaded_vehicle_ids": self._safe(traci.simulation.getLoadedIDList, default=[]),
            "loaded_vehicle_count": self._safe(traci.simulation.getLoadedNumber),
            "departed_vehicle_ids": self._safe(traci.simulation.getDepartedIDList, default=[]),
            "departed_vehicle_count": self._safe(traci.simulation.getDepartedNumber),
            "arrived_vehicle_ids": self._safe(traci.simulation.getArrivedIDList, default=[]),
            "arrived_vehicle_count": self._safe(traci.simulation.getArrivedNumber),
            "starting_teleport_ids": self._safe(traci.simulation.getStartingTeleportIDList, default=[]),
            "starting_teleport_count": self._safe(traci.simulation.getStartingTeleportNumber),
            "ending_teleport_ids": self._safe(traci.simulation.getEndingTeleportIDList, default=[]),
            "ending_teleport_count": self._safe(traci.simulation.getEndingTeleportNumber),
            "colliding_vehicle_ids": self._safe(traci.simulation.getCollidingVehiclesIDList, default=[]),
            "colliding_vehicle_count": self._safe(traci.simulation.getCollidingVehiclesNumber),
            "net_boundary": self._safe(traci.simulation.getNetBoundary),
            "simulation_step_counter": _UNAVAILABLE,  # not a direct TraCI getter
            "simulation_state": _UNAVAILABLE,          # not a single direct TraCI getter
        }

    # ==================================================================
    # 2. VEHICLE-LEVEL RAW DATA
    # ==================================================================
    def get_vehicle_data(self, vehicle_id: str) -> Dict[str, Any]:
        """
        Uses (all traci.vehicle.* getters, one call each): getTypeID,
        getVehicleClass, getRouteID, getRoute, getRouteIndex, getRoadID,
        getLaneID, getLaneIndex, getLanePosition, getPosition,
        getPosition3D, getDistance, getSpeed, getMaxSpeed, getAllowedSpeed,
        getAcceleration, getAngle, getLateralLanePosition, getDeparture,
        getWaitingTime, getAccumulatedWaitingTime, getTimeLoss, getLength,
        getWidth, getHeight, getColor, getShapeClass, getSignals,
        getSpeedFactor, getSpeedMode, getLaneChangeMode, getLine, getVia,
        getLeader, getNextTLS.

        Returns one dict of raw attributes for a single vehicle. If the
        vehicle has disappeared between the ID list being fetched and this
        call running, individual fields fail safely and are marked
        UNAVAILABLE rather than raising.

        Why useful: this is the full raw per-vehicle snapshot the Data
        Collection Layer needs to later derive density, queueing, waiting
        statistics, emergency-vehicle priority, and turning-movement intent.
        Nothing here is computed -- every field is a direct TraCI value.
        """
        v = traci.vehicle
        return {
            # Identification
            "vehicle_id": vehicle_id,
            "type_id": self._safe(v.getTypeID, vehicle_id),
            "vehicle_class": self._safe(v.getVehicleClass, vehicle_id),
            "route_id": self._safe(v.getRouteID, vehicle_id),

            # Position
            "edge_id": self._safe(v.getRoadID, vehicle_id),
            "lane_id": self._safe(v.getLaneID, vehicle_id),
            "lane_index": self._safe(v.getLaneIndex, vehicle_id),
            "lane_position": self._safe(v.getLanePosition, vehicle_id),
            "position_xy": self._safe(v.getPosition, vehicle_id),
            "position_xyz": self._safe(v.getPosition3D, vehicle_id),
            "distance_traveled": self._safe(v.getDistance, vehicle_id),

            # Movement
            "speed": self._safe(v.getSpeed, vehicle_id),
            "max_speed": self._safe(v.getMaxSpeed, vehicle_id),
            "allowed_speed": self._safe(v.getAllowedSpeed, vehicle_id),
            "acceleration": self._safe(v.getAcceleration, vehicle_id),
            "angle": self._safe(v.getAngle, vehicle_id),
            "lateral_lane_position": self._safe(v.getLateralLanePosition, vehicle_id),
            "current_direction_note": (
                "Not separately exposed by TraCI; use 'angle' (heading in "
                "degrees, 0=North clockwise) as the raw directional value."
            ),

            # Timing
            "departure_time": self._safe(v.getDeparture, vehicle_id),
            "waiting_time": self._safe(v.getWaitingTime, vehicle_id),
            "accumulated_waiting_time": self._safe(v.getAccumulatedWaitingTime, vehicle_id),
            "time_loss": self._safe(v.getTimeLoss, vehicle_id),

            # Route
            "route": self._safe(v.getRoute, vehicle_id, default=[]),
            "route_index": self._safe(v.getRouteIndex, vehicle_id),
            "remaining_route_note": (
                "Not directly exposed by TraCI; Data Collection Layer can "
                "derive it downstream as route[route_index:] using the raw "
                "'route' and 'route_index' fields above."
            ),
            "next_edge_note": (
                "No direct getNextEdge() in TraCI; derivable downstream as "
                "route[route_index + 1] using the raw 'route'/'route_index' "
                "fields above."
            ),

            # Vehicle properties
            "length": self._safe(v.getLength, vehicle_id),
            "width": self._safe(v.getWidth, vehicle_id),
            "height": self._safe(v.getHeight, vehicle_id),
            "color": self._safe(v.getColor, vehicle_id),
            "shape_class": self._safe(v.getShapeClass, vehicle_id),
            "signals": self._safe(v.getSignals, vehicle_id),
            "speed_factor": self._safe(v.getSpeedFactor, vehicle_id),
            "speed_mode": self._safe(v.getSpeedMode, vehicle_id),
            "lane_change_mode": self._safe(v.getLaneChangeMode, vehicle_id),
            "line": self._safe(v.getLine, vehicle_id),
            "via": self._safe(v.getVia, vehicle_id, default=[]),

            # Extra raw fields useful for signal control / emergency detection
            "leader": self._safe(v.getLeader, vehicle_id),   # (leader_id, gap) or None
            "next_traffic_lights": self._safe(v.getNextTLS, vehicle_id, default=[]),
        }

    def get_all_vehicle_data(self) -> List[Dict[str, Any]]:
        """
        Uses: traci.vehicle.getIDList(), then get_vehicle_data() per ID.
        Returns a list of raw per-vehicle dicts for every vehicle currently
        active in the network. Vehicles that disappear mid-loop (arrived,
        teleported out, collided) are skipped safely rather than raising.
        """
        results = []
        for vid in self._safe(traci.vehicle.getIDList, default=[]):
            try:
                results.append(self.get_vehicle_data(vid))
            except (TraCIException, FatalTraCIError):
                logger.debug("Vehicle %s disappeared mid-collection; skipping.", vid)
                continue
        return results

    # ==================================================================
    # 3. LANE-LEVEL RAW DATA
    # ==================================================================
    def get_lane_data(self, lane_id: str) -> Dict[str, Any]:
        """
        Uses (traci.lane.* getters): getEdgeID, getLength, getMaxSpeed,
        getWidth, getShape, getAllowed, getDisallowed,
        getLastStepVehicleNumber, getLastStepVehicleIDs, getLastStepMeanSpeed,
        getLastStepOccupancy, getLastStepHaltingNumber, getLastStepLength,
        getTraveltime.

        Returns one dict of raw attributes for a single lane.

        Note: TraCI does not separately expose a "waiting vehicle count" for
        lanes distinct from halting vehicles; getLastStepHaltingNumber
        (vehicles with speed < 0.1 m/s) is the closest raw proxy TraCI
        provides, and is returned as-is without reinterpretation.
        """
        l = traci.lane
        return {
            "lane_id": lane_id,
            "edge_id": self._safe(l.getEdgeID, lane_id),
            "length": self._safe(l.getLength, lane_id),
            "max_speed": self._safe(l.getMaxSpeed, lane_id),
            "width": self._safe(l.getWidth, lane_id),
            "shape": self._safe(l.getShape, lane_id),
            "allowed_vehicle_classes": self._safe(l.getAllowed, lane_id, default=[]),
            "disallowed_vehicle_classes": self._safe(l.getDisallowed, lane_id, default=[]),
            "vehicle_count": self._safe(l.getLastStepVehicleNumber, lane_id),
            "vehicle_ids": self._safe(l.getLastStepVehicleIDs, lane_id, default=[]),
            "mean_speed": self._safe(l.getLastStepMeanSpeed, lane_id),
            "occupancy_pct": self._safe(l.getLastStepOccupancy, lane_id),
            "halting_vehicle_count": self._safe(l.getLastStepHaltingNumber, lane_id),
            "waiting_vehicle_count_note": (
                "Not separately exposed by TraCI; 'halting_vehicle_count' "
                "(speed < 0.1 m/s) is the closest raw proxy TraCI provides."
            ),
            "mean_vehicle_length": self._safe(l.getLastStepLength, lane_id),
            "travel_time": self._safe(l.getTraveltime, lane_id),
        }

    def get_all_lane_data(self) -> List[Dict[str, Any]]:
        """Uses traci.lane.getIDList(), then get_lane_data() per lane. Skips internal junction lanes (IDs starting with ':')."""
        results = []
        for lane_id in self._safe(traci.lane.getIDList, default=[]):
            if lane_id.startswith(":"):
                continue
            results.append(self.get_lane_data(lane_id))
        return results

    def _get_lane_heading(self, lane_id: str) -> float:
        """
        Perf helper (cached): computes a lane's compass heading once from
        traci.lane.getShape(), reusing the cached value on later calls
        instead of recomputing it every step. Not used for any traffic
        metric -- only to label turning movements with a compass direction.
        """
        if lane_id in self._lane_angle_cache:
            return self._lane_angle_cache[lane_id]
        shape = self._safe(traci.lane.getShape, lane_id, default=None)
        angle = 0.0
        if shape and len(shape) >= 2:
            (x1, y1), (x2, y2) = shape[0], shape[-1]
            angle = math.degrees(math.atan2(x2 - x1, y2 - y1)) % 360
        self._lane_angle_cache[lane_id] = angle
        return angle

    # ==================================================================
    # 4. EDGE-LEVEL RAW DATA
    # ==================================================================
    def get_edge_data(self, edge_id: str) -> Dict[str, Any]:
        """
        Uses (traci.edge.* getters): getLastStepVehicleNumber,
        getLastStepVehicleIDs, getLastStepMeanSpeed, getTraveltime,
        getWaitingTime, getLastStepOccupancy, getCO2Emission, getCOEmission,
        getHCEmission, getPMxEmission, getNOxEmission, getFuelConsumption,
        getNoiseEmission, getElectricityConsumption, getLaneNumber.

        These emission/fuel/noise/waiting-time values are SUMO's own
        internally computed instantaneous readings for the current step,
        retrieved as-is -- this file does not total, average, or otherwise
        process them.

        Note: TraCI's Edge domain does not expose edge length directly.
        The raw source for length is the lane domain (each edge's lanes
        share the same length) -- e.g. lane getLength() on "<edge_id>_0".
        That value is returned by get_lane_data(), not duplicated here.
        """
        e = traci.edge
        return {
            "edge_id": edge_id,
            "vehicle_count": self._safe(e.getLastStepVehicleNumber, edge_id),
            "vehicle_ids": self._safe(e.getLastStepVehicleIDs, edge_id, default=[]),
            "mean_speed": self._safe(e.getLastStepMeanSpeed, edge_id),
            "travel_time": self._safe(e.getTraveltime, edge_id),
            "waiting_time": self._safe(e.getWaitingTime, edge_id),
            "occupancy_pct": self._safe(e.getLastStepOccupancy, edge_id),
            "co2_emission": self._safe(e.getCO2Emission, edge_id),
            "co_emission": self._safe(e.getCOEmission, edge_id),
            "hc_emission": self._safe(e.getHCEmission, edge_id),
            "pmx_emission": self._safe(e.getPMxEmission, edge_id),
            "nox_emission": self._safe(e.getNOxEmission, edge_id),
            "fuel_consumption": self._safe(e.getFuelConsumption, edge_id),
            "noise_emission": self._safe(e.getNoiseEmission, edge_id),
            "electricity_consumption": self._safe(e.getElectricityConsumption, edge_id),
            "lane_number": self._safe(e.getLaneNumber, edge_id),
            "length_note": (
                "Not directly exposed by TraCI Edge domain; use lane-level "
                "length from get_lane_data() on this edge's lanes "
                "('<edge_id>_0', '<edge_id>_1', ...)."
            ),
        }

    def get_all_edge_data(self) -> List[Dict[str, Any]]:
        """Uses traci.edge.getIDList(), then get_edge_data() per edge. Skips internal junction edges (IDs starting with ':')."""
        results = []
        for edge_id in self._safe(traci.edge.getIDList, default=[]):
            if edge_id.startswith(":"):
                continue
            results.append(self.get_edge_data(edge_id))
        return results

    # ==================================================================
    # 5. JUNCTION-LEVEL RAW DATA
    # ==================================================================
    def get_junction_data(self, junction_id: str) -> Dict[str, Any]:
        """
        Uses (traci.junction.* getters): getPosition, getShape.

        Note: the installed TraCI Junction domain does not expose a
        getPosition3D() method (only getPosition(), 2D) -- calling it would
        raise AttributeError before _safe()'s try/except can even catch it,
        since the attribute lookup itself fails at call-site evaluation.
        Per the project rule of never calling a TraCI method that doesn't
        exist in the installed version, "position_xyz" is explicitly marked
        unavailable rather than invented or approximated.

        TraCI's Junction domain also does NOT expose incoming edges,
        outgoing edges, connected lanes, or the controlling traffic light
        directly. Building that mapping requires parsing the SUMO network
        file (e.g. via the sumolib library), which is outside the TraCI API
        and therefore outside this interface's scope. Those fields are
        marked unavailable rather than approximated.
        """
        j = traci.junction
        return {
            "junction_id": junction_id,
            "position_xy": self._safe(j.getPosition, junction_id),
            "position_xyz": _UNAVAILABLE,  # traci.junction.getPosition3D() does not exist
            "shape": self._safe(j.getShape, junction_id),
            "incoming_edges": _UNAVAILABLE,          # not exposed via TraCI Junction domain
            "outgoing_edges": _UNAVAILABLE,          # not exposed via TraCI Junction domain
            "connected_lanes": _UNAVAILABLE,         # not exposed via TraCI Junction domain
            "controlled_traffic_light": _UNAVAILABLE,  # not exposed via TraCI Junction domain
            "unavailable_fields_note": (
                "incoming_edges / outgoing_edges / connected_lanes / "
                "controlled_traffic_light require parsing the SUMO network "
                "file (e.g. via sumolib) -- TraCI's Junction domain does not "
                "provide them directly."
            ),
        }

    def get_all_junction_data(self) -> List[Dict[str, Any]]:
        """
        Uses traci.junction.getIDList(), then get_junction_data() per
        junction. Skips internal junctions (IDs starting with ':'), same as
        get_all_lane_data() / get_all_edge_data() already do: internal
        junctions are SUMO-generated intersection-geometry helpers with
        degenerate (often empty) shapes that this TraCI version's
        junction.getShape() cannot parse (raises a low-level struct-unpack
        error rather than a catchable TraCIException/AttributeError), so
        they are excluded rather than crashing the whole snapshot collection.
        """
        results = []
        for jid in self._safe(traci.junction.getIDList, default=[]):
            if jid.startswith(":"):
                continue
            results.append(self.get_junction_data(jid))
        return results

    # ==================================================================
    # 6. TRAFFIC LIGHT RAW DATA
    # ==================================================================
    def get_traffic_light_data(self, tls_id: str) -> Dict[str, Any]:
        """
        Uses (traci.trafficlight.* getters): getRedYellowGreenState,
        getPhase, getPhaseName, getPhaseDuration, getNextSwitch, getProgram,
        getControlledLanes, getControlledLinks, getAllProgramLogics.

        getAllProgramLogics returns SUMO's own native Logic objects
        (raw program/phase definitions) unmodified -- this interface does
        not parse or summarize them, just passes them through.

        getPhaseName may not exist on older SUMO versions; handled safely.
        """
        t = traci.trafficlight
        return {
            "traffic_light_id": tls_id,
            "current_state": self._safe(t.getRedYellowGreenState, tls_id),
            "current_phase_index": self._safe(t.getPhase, tls_id),
            "current_phase_name": self._safe(t.getPhaseName, tls_id),
            "phase_duration": self._safe(t.getPhaseDuration, tls_id),
            "next_switch_time": self._safe(t.getNextSwitch, tls_id),
            "current_program": self._safe(t.getProgram, tls_id),
            "controlled_lanes": self._safe(t.getControlledLanes, tls_id, default=[]),
            "controlled_links": self._safe(t.getControlledLinks, tls_id, default=[]),
            "program_logics_raw": self._safe(t.getAllProgramLogics, tls_id, default=[]),
        }

    def get_all_traffic_light_data(self) -> List[Dict[str, Any]]:
        """Uses traci.trafficlight.getIDList(), then get_traffic_light_data() per TLS."""
        return [
            self.get_traffic_light_data(tid)
            for tid in self._safe(traci.trafficlight.getIDList, default=[])
        ]

    # ==================================================================
    # 7. TURNING MOVEMENT RAW DATA (existing implementation, extended)
    # ==================================================================
    def get_turning_movements(self) -> List[Dict[str, Any]]:
        """
        Uses: traci.lane.getLinks(lane_id) for each non-internal lane, plus
        traci.lane.getAllowed()/getDisallowed() on the approached (target)
        lane, and this module's own compass-heading helper (derived purely
        from lane geometry, not simulation state).

        traci.lane.getLinks() returns, per outgoing connection, an 8-tuple:
        (approachedLane, hasPrio, isOpen, hasFoe, approachedInternalLane,
        state, direction, length). 'direction' is the raw turn code this
        method maps to a readable label via _TRACI_DIRECTION_MAP (a lookup
        table only -- no computation): 's'=straight, 'l'=left, 'r'=right,
        'L'=partial-left, 'R'=partial-right, 't'=u-turn. 'state' is a
        SEPARATE raw single-character link-state code (e.g. 'G'/'g'/'r'/
        'y'/'R'/'s'/'u'/'o'/'O'/'c'/'C'/'-'/'='/'m'/'M') describing the
        link's current/static control state -- it is NOT a boolean, and is
        returned as-is under "link_state_raw" without this interface
        interpreting it. This method also returns the raw allowed/
        disallowed vehicle classes of the target lane, so the Data
        Collection Layer can determine allowed left/right/straight/u-turn
        movements without this interface making that determination itself.

        This does NOT determine a vehicle's intended movement -- for that,
        the Data Collection Layer should combine this with the raw
        'route' / 'route_index' fields from get_vehicle_data().
        """
        movements = []
        for lane_id in self._safe(traci.lane.getIDList, default=[]):
            if lane_id.startswith(":"):
                continue
            links = self._safe(traci.lane.getLinks, lane_id, default=[])
            if not links:
                continue

            heading = angle_to_compass(self._get_lane_heading(lane_id))

            for link in links:
                approached_lane = link[0] if len(link) > 0 else None
                direction_code = link[6] if len(link) > 6 else "invalid"
                turn_label = _TRACI_DIRECTION_MAP.get(direction_code, "unknown")

                movements.append({
                    "lane_id": lane_id,
                    "approached_lane": approached_lane,
                    "approach_heading": heading,
                    "turn_code_raw": direction_code,
                    "turn_label": turn_label,
                    "movement": f"{heading}-{turn_label}",
                    "has_priority": link[1] if len(link) > 1 else _UNAVAILABLE,
                    "is_open": link[2] if len(link) > 2 else _UNAVAILABLE,
                    "has_foe": link[3] if len(link) > 3 else _UNAVAILABLE,
                    "approached_internal_lane": link[4] if len(link) > 4 else _UNAVAILABLE,
                    "link_state_raw": link[5] if len(link) > 5 else _UNAVAILABLE,
                    "link_state_note": (
                        "Raw single-character SUMO link-state code (not a "
                        "boolean): e.g. 'G'/'g' green, 'r'/'R' red, 'y'/'Y' "
                        "yellow, 's' stop, 'u' allway_stop, 'o' controller "
                        "off (flashing), 'O' controller off, 'c'/'C' "
                        "conflict-yield/pass, '-' dead end, '=' equal, "
                        "'m' minor link, 'M' major link. Returned as-is; "
                        "this interface does not interpret it."
                    ),
                    "connection_length": link[7] if len(link) > 7 else _UNAVAILABLE,
                    "target_lane_allowed_classes": (
                        self._safe(traci.lane.getAllowed, approached_lane, default=[])
                        if approached_lane else []
                    ),
                    "target_lane_disallowed_classes": (
                        self._safe(traci.lane.getDisallowed, approached_lane, default=[])
                        if approached_lane else []
                    ),
                })
        return movements

    # ==================================================================
    # 9. NETWORK-LEVEL ID HELPERS
    # ==================================================================
    def get_all_vehicle_ids(self) -> List[str]:
        """Uses traci.vehicle.getIDList()."""
        return self._safe(traci.vehicle.getIDList, default=[])

    def get_all_lane_ids(self) -> List[str]:
        """Uses traci.lane.getIDList()."""
        return self._safe(traci.lane.getIDList, default=[])

    def get_all_edge_ids(self) -> List[str]:
        """Uses traci.edge.getIDList()."""
        return self._safe(traci.edge.getIDList, default=[])

    def get_all_junction_ids(self) -> List[str]:
        """Uses traci.junction.getIDList()."""
        return self._safe(traci.junction.getIDList, default=[])

    def get_all_traffic_light_ids(self) -> List[str]:
        """Uses traci.trafficlight.getIDList()."""
        return self._safe(traci.trafficlight.getIDList, default=[])

    def get_all_route_ids(self) -> List[str]:
        """Uses traci.route.getIDList()."""
        return self._safe(traci.route.getIDList, default=[])

    def get_all_detector_ids(self) -> Dict[str, List[str]]:
        """
        Uses traci.inductionloop.getIDList() (E1), traci.lanearea.getIDList()
        (E2), traci.multientryexit.getIDList() (E3). Returns an empty list
        per type if that detector type isn't defined in the network -- SUMO
        itself returns [] in that case rather than raising.
        """
        return {
            "induction_loop_e1": self._safe(traci.inductionloop.getIDList, default=[]),
            "lane_area_e2": self._safe(traci.lanearea.getIDList, default=[]),
            "multi_entry_exit_e3": self._safe(traci.multientryexit.getIDList, default=[]),
        }

    def get_route_edges(self, route_id: str) -> List[str]:
        """Uses traci.route.getEdges(route_id). Returns the raw edge list for a defined route."""
        return self._safe(traci.route.getEdges, route_id, default=[])

    # ==================================================================
    # SNAPSHOT BUNDLER -- the single hand-off point to the Data Collection
    # Layer. This does not compute or aggregate anything; it just calls the
    # get_all_*_data() methods above (each already a pure raw retrieval) and
    # packages their outputs together under one dict so the Data Collection
    # Layer's update() has one call to make per simulation step.
    # ==================================================================
    def collect_step_data(self) -> Dict[str, Any]:
        """
        Intended caller: data_collection.py's update() method, once per step:
            snapshot = traci_api.collect_step_data()
            data_collection_layer.update(snapshot)

        Returns a dict with keys: "simulation", "vehicles", "lanes", "edges",
        "junctions", "traffic_lights", "turning_movements" -- each value is
        exactly what the corresponding get_* method above already returns
        when its TraCIConfig.collect_* toggle is True (the default), and an
        empty list when disabled (see TraCIConfig). No new values are
        produced here; this only wraps existing raw results together for a
        single hand-off. Skipping a domain never changes the shape of this
        dict, only whether that domain's list is populated or empty, so
        downstream code that already guards on "or []" (see
        data_collection.py's update()) needs no changes either way.
        """
        return {
            "simulation": self.get_simulation_data(),
            "vehicles": self.get_all_vehicle_data(),
            "lanes": self.get_all_lane_data(),
            "edges": self.get_all_edge_data() if self.config.collect_edges else [],
            "junctions": self.get_all_junction_data() if self.config.collect_junctions else [],
            "traffic_lights": (
                self.get_all_traffic_light_data() if self.config.collect_traffic_lights else []
            ),
            "turning_movements": (
                self.get_turning_movements() if self.config.collect_turning_movements else []
            ),
        }