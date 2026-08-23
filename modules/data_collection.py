# 1. Connect to SUMO
# 2. Advance the simulation
#3. Collect data from the simulation(traci)
# Arrival Time
 #Vehicle ID	
 #Time when vehicle entered the road/intersection
#Queue Length
 #Vehicle ID
 #Vehicle position
 #Vehicle speed
 #Lane/edge ID
#PCU-weighted Density
 #Vehicle type
 #Vehicle count
 #Road length (or capacity)
# 4.Detect emergency vehicles and their approach direction
# 5. Save the collected data
"""
data_collection.py
===================

Data Collection Layer of the Adaptive Alternate Green Wave Traffic Signal
Management System.

    SUMO -> TraCI API -> traci_interface.py -> data_collection.py
                                                    |
                                    ,---------------+----------------.
                              Decision Engine   Emergency Module   ...

This is the ONLY module (besides traci_interface.py itself) allowed to sit
between SUMO and the rest of the system. The Decision Engine, Emergency
Vehicle Module, QUBO Builder, QAOA Solver, and Signal Controller must all
get their traffic information from THIS file -- never from traci_interface.py
directly, and never from SUMO/TraCI directly.

Responsibilities of this class (and ONLY these):
    1. Receive the raw snapshot from traci_interface.collect_step_data()
       once per simulation step.
    2. Store the latest snapshot.
    3. Store the previous snapshot.
    4. Organize the raw lists into id-keyed dictionaries for O(1) lookup.
    5. Build lookup indexes (grouping raw data by lane/edge/vehicle), again
       without computing anything new.
    6. Provide getter methods for every other module to read through.
    7. Never call traci / SUMO directly.
    8. Never calculate density, PCU-weighted density, queue length, waiting
       averages, congestion, green time, signal priority, emergency
       priority, QUBO inputs, or any other derived metric. All of that is
       out of scope for this file and belongs to later modules.

Every dictionary and index below is built purely by re-keying or grouping
values that traci_interface.py already returned -- no arithmetic, no
thresholds, no classification beyond reading an already-raw field.
"""

import logging
from typing import Any, Dict, List, Optional

# Only imported for its raw signal-bit reference table (a lookup dict
# documenting what TraCI's own vehicle.getSignals() bitmask means) --
# this import does NOT call traci or SUMO in any way.
#
# Relative import when this file is used normally (as modules/data_collection.py,
# part of the `modules` package), with a fallback to a plain absolute import so
# the __main__ demo below can still run this file directly as a script
# (`python data_collection.py`) without crashing with
# "ImportError: attempted relative import with no known parent package".
try:
    from .traci_interface import VEHICLE_SIGNAL_BIT_REFERENCE
except ImportError:
    from traci_interface import VEHICLE_SIGNAL_BIT_REFERENCE

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("data_collection")

# Bit position (from VEHICLE_SIGNAL_BIT_REFERENCE) that SUMO itself defines
# as the emergency blue light signal. Used only to IDENTIFY which vehicles
# already carry that raw flag -- not to assign or score priority.
_EMERGENCY_BLUE_LIGHT_BIT = next(
    (bit for bit, name in VEHICLE_SIGNAL_BIT_REFERENCE.items()
     if name == "emergency_blue_light"),
    None,
)
if _EMERGENCY_BLUE_LIGHT_BIT is None:
    raise RuntimeError(
        "data_collection.py could not find an 'emergency_blue_light' entry in "
        "traci_interface.VEHICLE_SIGNAL_BIT_REFERENCE. This mapping is required "
        "to identify emergency vehicles via their raw TraCI signal bitmask -- "
        "check that traci_interface.py still defines this bit."
    )
_EMERGENCY_BLUE_LIGHT_MASK = 1 << _EMERGENCY_BLUE_LIGHT_BIT

_UNAVAILABLE_MAPPING_NOTE = (
    "Empty: requires junction<->edge/lane connectivity, which TraCI's "
    "Junction domain does not expose (see traci_interface.get_junction_data()). "
    "Will stay empty unless that mapping is supplied to this layer from an "
    "external source (e.g. network-file parsing done outside TraCI)."
)


class DataCollectionLayer:
    """
    Single source of truth for raw traffic-network state.

    Storage dictionaries (all keyed by the entity's own raw ID):
        simulation_data         -- raw simulation-level counters/lists
        vehicles_by_id          -- vehicle_id -> raw vehicle dict
        lanes_by_id             -- lane_id -> raw lane dict
        edges_by_id             -- edge_id -> raw edge dict
        junctions_by_id         -- junction_id -> raw junction dict
        traffic_lights_by_id    -- tls_id -> raw traffic light dict
        turning_data            -- lane_id -> list of raw turning-movement dicts
        emergency_vehicles      -- vehicle_id -> raw vehicle dict, restricted
                                    to vehicles whose raw vehicle_class or
                                    signals field already marks them as
                                    emergency per SUMO's own definitions

    Lookup indexes (grouped from the dictionaries above, no computation):
        vehicles_by_lane, vehicles_by_edge, vehicles_by_junction,
        lanes_by_edge, lanes_by_junction, edges_by_junction,
        traffic_lights_by_junction, routes_by_vehicle, emergency_vehicle_ids

    Expected consumers: Decision Engine, Emergency Vehicle Module, QUBO
    Builder, QAOA Solver, Signal Controller -- all read-only, through the
    getter methods below.
    """

    def __init__(self) -> None:
        # --- current snapshot storage -----------------------------------
        self.simulation_data: Dict[str, Any] = {}
        self.vehicles_by_id: Dict[str, Dict[str, Any]] = {}
        self.lanes_by_id: Dict[str, Dict[str, Any]] = {}
        self.edges_by_id: Dict[str, Dict[str, Any]] = {}
        self.junctions_by_id: Dict[str, Dict[str, Any]] = {}
        self.traffic_lights_by_id: Dict[str, Dict[str, Any]] = {}
        self.turning_data: Dict[str, List[Dict[str, Any]]] = {}
        self.emergency_vehicles: Dict[str, Dict[str, Any]] = {}

        # --- raw snapshot references (whole-dict, not duplicated field by
        # field, to avoid keeping two copies of the same data in memory) ---
        self._current_snapshot: Optional[Dict[str, Any]] = None
        self._previous_snapshot: Optional[Dict[str, Any]] = None

        # --- lookup indexes ----------------------------------------------
        self.vehicles_by_lane: Dict[str, List[str]] = {}
        self.vehicles_by_edge: Dict[str, List[str]] = {}
        self.vehicles_by_junction: Dict[str, List[str]] = {}
        self.lanes_by_edge: Dict[str, List[str]] = {}
        self.lanes_by_junction: Dict[str, List[str]] = {}
        self.edges_by_junction: Dict[str, List[str]] = {}
        self.traffic_lights_by_junction: Dict[str, List[str]] = {}
        self.routes_by_vehicle: Dict[str, List[str]] = {}
        self.emergency_vehicle_ids: List[str] = []

    # ======================================================================
    # UPDATE -- the single write path into this layer
    # ======================================================================
    def update(self, snapshot: Dict[str, Any]) -> None:
        """
        Called once per simulation step with the dict returned by
        traci_interface.collect_step_data(), i.e.:
            {
              "simulation": {...},
              "vehicles": [ {...}, ... ],
              "lanes": [ {...}, ... ],
              "edges": [ {...}, ... ],
              "junctions": [ {...}, ... ],
              "traffic_lights": [ {...}, ... ],
              "turning_movements": [ {...}, ... ],
            }

        Steps performed (re-keying/grouping only, no calculation):
            1. Shift the current snapshot into "previous".
            2. Store the new snapshot reference.
            3. Re-key each raw list into an id -> dict lookup table.
            4. Group turning movements by their originating lane.
            5. Identify (not score) emergency vehicles via SUMO's own raw
               vehicle_class / signals fields.
            6. Rebuild all lookup indexes from the freshly stored data.
        """
        previous_snapshot = self._current_snapshot

        simulation_data = snapshot.get("simulation", {}) or {}

        vehicles_by_id = {
            v["vehicle_id"]: v for v in snapshot.get("vehicles", []) or []
            if v.get("vehicle_id")
        }
        lanes_by_id = {
            l["lane_id"]: l for l in snapshot.get("lanes", []) or []
            if l.get("lane_id")
        }
        edges_by_id = {
            e["edge_id"]: e for e in snapshot.get("edges", []) or []
            if e.get("edge_id")
        }
        junctions_by_id = {
            j["junction_id"]: j for j in snapshot.get("junctions", []) or []
            if j.get("junction_id")
        }
        traffic_lights_by_id = {
            t["traffic_light_id"]: t for t in snapshot.get("traffic_lights", []) or []
            if t.get("traffic_light_id")
        }

        turning_data: Dict[str, List[Dict[str, Any]]] = {}
        for movement in snapshot.get("turning_movements", []) or []:
            lane_id = movement.get("lane_id")
            if lane_id is None:
                continue
            turning_data.setdefault(lane_id, []).append(movement)

        emergency_vehicles = {
            vid: vdata for vid, vdata in vehicles_by_id.items()
            if self._is_flagged_emergency(vdata)
        }

        indexes = self._build_indexes(vehicles_by_id, lanes_by_id, emergency_vehicles)

        # Single, tight commit block: everything a getter can read is
        # assigned together here rather than being spread across the whole
        # method, minimizing the window in which a concurrent reader (see
        # main_controller.py's background decision-cycle thread) could
        # observe a mix of this step's and the previous step's data.
        self._previous_snapshot = previous_snapshot
        self._current_snapshot = snapshot
        self.simulation_data = simulation_data
        self.vehicles_by_id = vehicles_by_id
        self.lanes_by_id = lanes_by_id
        self.edges_by_id = edges_by_id
        self.junctions_by_id = junctions_by_id
        self.traffic_lights_by_id = traffic_lights_by_id
        self.turning_data = turning_data
        self.emergency_vehicles = emergency_vehicles
        (
            self.vehicles_by_lane,
            self.vehicles_by_edge,
            self.routes_by_vehicle,
            self.lanes_by_edge,
            self.emergency_vehicle_ids,
            self.vehicles_by_junction,
            self.lanes_by_junction,
            self.edges_by_junction,
            self.traffic_lights_by_junction,
        ) = indexes

    @staticmethod
    def _is_flagged_emergency(vehicle_data: Dict[str, Any]) -> bool:
        """
        Identification only -- reads two raw fields already present on the
        vehicle dict and checks them against SUMO's own definitions:
          - vehicle_class == "emergency" (SUMO's built-in vClass), OR
          - the raw 'signals' bitmask has the emergency_blue_light bit set
            (bit position taken from VEHICLE_SIGNAL_BIT_REFERENCE, which is
            TraCI's own documented bit meaning, not a value invented here).
        This assigns no priority, score, or ranking -- it only tells later
        modules which vehicles already carry an emergency flag from SUMO.
        """
        if vehicle_data.get("vehicle_class") == "emergency":
            return True
        signals = vehicle_data.get("signals")
        if isinstance(signals, int) and signals & _EMERGENCY_BLUE_LIGHT_MASK:
            return True
        return False

    @staticmethod
    def _build_indexes(
        vehicles_by_id: Dict[str, Dict[str, Any]],
        lanes_by_id: Dict[str, Dict[str, Any]],
        emergency_vehicles: Dict[str, Dict[str, Any]],
    ):
        """
        Computes every lookup index from the dictionaries update() just
        built, purely by grouping existing raw fields -- no new values are
        produced. Returns plain local values rather than mutating self.*
        directly so update() can commit everything in one final block (see
        the comment there).
        """
        vehicles_by_lane: Dict[str, List[str]] = {}
        vehicles_by_edge: Dict[str, List[str]] = {}
        routes_by_vehicle: Dict[str, List[str]] = {}
        for vid, vdata in vehicles_by_id.items():
            lane_id = vdata.get("lane_id")
            if lane_id:
                vehicles_by_lane.setdefault(lane_id, []).append(vid)

            edge_id = vdata.get("edge_id")
            if edge_id:
                vehicles_by_edge.setdefault(edge_id, []).append(vid)

            route = vdata.get("route")
            if route:
                routes_by_vehicle[vid] = route

        lanes_by_edge: Dict[str, List[str]] = {}
        for lane_id, ldata in lanes_by_id.items():
            edge_id = ldata.get("edge_id")
            if edge_id:
                lanes_by_edge.setdefault(edge_id, []).append(lane_id)

        emergency_vehicle_ids = list(emergency_vehicles.keys())

        # These require junction<->edge/lane connectivity that TraCI's
        # Junction domain does not provide (see traci_interface.py's
        # get_junction_data() notes). Left empty and documented rather than
        # guessed at. If an upstream module ever supplies that mapping,
        # this is where it would be grouped in.
        vehicles_by_junction: Dict[str, List[str]] = {}
        lanes_by_junction: Dict[str, List[str]] = {}
        edges_by_junction: Dict[str, List[str]] = {}
        traffic_lights_by_junction: Dict[str, List[str]] = {}

        return (
            vehicles_by_lane,
            vehicles_by_edge,
            routes_by_vehicle,
            lanes_by_edge,
            emergency_vehicle_ids,
            vehicles_by_junction,
            lanes_by_junction,
            edges_by_junction,
            traffic_lights_by_junction,
        )

    # ======================================================================
    # GETTERS -- read-only access for every other module
    # ======================================================================
    def get_simulation_data(self) -> Dict[str, Any]:
        """Returns the raw simulation-level dict stored by the latest update()."""
        return self.simulation_data

    def get_previous_simulation_data(self) -> Dict[str, Any]:
        """Returns the raw simulation-level dict from the snapshot before the latest one (or {} if only one update has occurred)."""
        if self._previous_snapshot is None:
            return {}
        return self._previous_snapshot.get("simulation", {}) or {}

    def get_vehicle(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Returns the raw vehicle dict for one vehicle_id, or None if it isn't in the current snapshot."""
        return self.vehicles_by_id.get(vehicle_id)

    def get_all_vehicles(self) -> List[Dict[str, Any]]:
        """Returns every raw vehicle dict in the current snapshot."""
        return list(self.vehicles_by_id.values())

    def get_lane(self, lane_id: str) -> Optional[Dict[str, Any]]:
        """Returns the raw lane dict for one lane_id, or None if not present."""
        return self.lanes_by_id.get(lane_id)

    def get_all_lanes(self) -> List[Dict[str, Any]]:
        """Returns every raw lane dict in the current snapshot."""
        return list(self.lanes_by_id.values())

    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Returns the raw edge dict for one edge_id, or None if not present."""
        return self.edges_by_id.get(edge_id)

    def get_all_edges(self) -> List[Dict[str, Any]]:
        """Returns every raw edge dict in the current snapshot."""
        return list(self.edges_by_id.values())

    def get_junction(self, junction_id: str) -> Optional[Dict[str, Any]]:
        """Returns the raw junction dict for one junction_id, or None if not present."""
        return self.junctions_by_id.get(junction_id)

    def get_all_junctions(self) -> List[Dict[str, Any]]:
        """Returns every raw junction dict in the current snapshot."""
        return list(self.junctions_by_id.values())

    def get_traffic_light(self, tls_id: str) -> Optional[Dict[str, Any]]:
        """Returns the raw traffic light dict for one tls_id, or None if not present."""
        return self.traffic_lights_by_id.get(tls_id)

    def get_all_traffic_lights(self) -> List[Dict[str, Any]]:
        """Returns every raw traffic light dict in the current snapshot."""
        return list(self.traffic_lights_by_id.values())

    def get_turning_data(self, lane_id: Optional[str] = None) -> Any:
        """
        With no argument: returns the full lane_id -> [raw turning movement
        dicts] mapping. With a lane_id: returns just that lane's list of raw
        turning-movement dicts (or [] if the lane has none recorded).
        """
        if lane_id is not None:
            return self.turning_data.get(lane_id, [])
        return self.turning_data

    def get_emergency_vehicles(self) -> List[Dict[str, Any]]:
        """Returns the raw vehicle dicts for every vehicle currently flagged as emergency (see _is_flagged_emergency)."""
        return list(self.emergency_vehicles.values())

    # --- additional lookup getters -----------------------------------
    def get_vehicles_in_lane(self, lane_id: str) -> List[Dict[str, Any]]:
        """Returns the raw vehicle dicts for every vehicle currently on the given lane (via the vehicles_by_lane index)."""
        return [self.vehicles_by_id[vid] for vid in self.vehicles_by_lane.get(lane_id, []) if vid in self.vehicles_by_id]

    def get_vehicles_on_edge(self, edge_id: str) -> List[Dict[str, Any]]:
        """Returns the raw vehicle dicts for every vehicle currently on the given edge (via the vehicles_by_edge index)."""
        return [self.vehicles_by_id[vid] for vid in self.vehicles_by_edge.get(edge_id, []) if vid in self.vehicles_by_id]

    def get_lanes_of_junction(self, junction_id: str) -> List[Dict[str, Any]]:
        """
        Returns the raw lane dicts connected to the given junction.
        Currently always [] -- see lanes_by_junction / _UNAVAILABLE_MAPPING_NOTE:
        TraCI does not expose junction<->lane connectivity directly.
        """
        return [self.lanes_by_id[lid] for lid in self.lanes_by_junction.get(junction_id, []) if lid in self.lanes_by_id]

    def get_edges_of_junction(self, junction_id: str) -> List[Dict[str, Any]]:
        """
        Returns the raw edge dicts connected to the given junction.
        Currently always [] -- see edges_by_junction / _UNAVAILABLE_MAPPING_NOTE:
        TraCI does not expose junction<->edge connectivity directly.
        """
        return [self.edges_by_id[eid] for eid in self.edges_by_junction.get(junction_id, []) if eid in self.edges_by_id]

    def get_controlled_lanes(self, tls_id: str) -> List[str]:
        """Returns the raw 'controlled_lanes' list already stored on that traffic light's dict."""
        tls = self.traffic_lights_by_id.get(tls_id)
        return tls.get("controlled_lanes", []) if tls else []

    def get_routes_of_vehicle(self, vehicle_id: str) -> List[str]:
        """Returns the raw route (list of edge IDs) for the given vehicle, via the routes_by_vehicle index."""
        return self.routes_by_vehicle.get(vehicle_id, [])


# ---------------------------------------------------------------------------
# Example of how this layer plugs into the rest of the pipeline.
# Illustrative only -- downstream modules (Decision Engine, Emergency
# Vehicle Module, QUBO Builder, QAOA Solver, Signal Controller) are not
# implemented here; they would each take a DataCollectionLayer instance and
# call its getters.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from traci_interface import TraCIInterface, TraCIConfig

    config = TraCIConfig(
        sumocfg_path="network/simulation.sumocfg",  # <-- point at your .sumocfg
        use_gui=False,
        step_length=1.0,
    )

    traci_api = TraCIInterface(config)
    data_layer = DataCollectionLayer()

    traci_api.start_simulation()
    try:
        while traci_api.is_running():
            traci_api.step()
            snapshot = traci_api.collect_step_data()
            data_layer.update(snapshot)

            # Downstream modules would now read through data_layer, e.g.:
            #   emergency_module.check(data_layer.get_emergency_vehicles())
            #   decision_engine.evaluate(data_layer.get_all_lanes())
            logger.info(
                "t=%s | vehicles=%d | emergency=%d",
                data_layer.get_simulation_data().get("current_time"),
                len(data_layer.get_all_vehicles()),
                len(data_layer.get_emergency_vehicles()),
            )
    finally:
        traci_api.close_simulation()