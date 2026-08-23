#For the calculation get all the inputs from data collection
# 1. Calculate wait time
 #Formula :
# 2. Calculate PCU-weighted density
 #Total PCU=∑(Vehicle Count×PCU Weight)
 #  Vehicle	PCU
 # Motorcycle	0.5
 # Car	1.0
 # Auto	1.2
 # Bus	3.0
 # Truck	3.0
 #Density=Total PCU​/Road Length
# 3. Calculate queue length
 #Formula: 
# 4. Return all calculated metrics
# 5. Store the calculated metrics in a dictionary or a data structure for further analysis or visualization
"""
Traffic Metric Calculation
===========================

Turns raw per-vehicle data from the Data Collection Layer into the
per-approach metrics the rest of the pipeline runs on: wait time,
PCU-weighted density, and queue length.

Flow implemented (matches the flowchart originally sketched at the top of
this file):

    1. Get all inputs from the Data Collection Layer (raw per-vehicle data).
    2. Calculate wait time per approach.
    3. Calculate PCU-weighted density per approach, normalized against
       jam capacity so it is a comparable ratio across approaches of
       different lengths (see section 11: "density percentage must be
       normalized consistently"):
           Total PCU      = sum(vehicle_count_of_type * PCU_weight)
           Jam PCU Capacity = Road Length / jam_spacing_per_pcu_m
           Density        = Total PCU / Jam PCU Capacity
    4. Calculate queue length per approach.
    5. Return all calculated metrics.
    6. Store the calculated metrics in a dictionary for further analysis
       or visualisation (`self.last_metrics`).

Interface
---------
    calculator.calculate(vehicle_data) -> dict

This is what feeds `current_metrics` into predictor.Predictor.predict(),
decision_engine.DecisionEngine.decide(), and emergency_preemption's
`check()` throughout the rest of the pipeline, so the output shape below is
load-bearing for the whole system:

    {
        "N": {
            "density": 0.42,          # normalized occupancy ratio (0=empty, 1.0=jam-packed)
            "queue_length": 6,        # count of currently-halted vehicles
            "wait_time": 38.5,        # worst-case: longest-waiting vehicle
            "avg_wait_time": 14.2,    # mean waiting time across the approach
            "vehicle_count": 11,      # total vehicles currently on the approach
        },
        "E": {...}, "S": {...}, "W": {...}
    }

Every direction in config.DIRECTIONS is always present in the output, even
if no vehicles are currently on that approach (all-zero metrics), so
downstream modules never have to guard against a missing key.

Expected `vehicle_data` shape (from data_collection.py)
---------------------------------------------------------
A list of per-vehicle dicts:

    {
        "id": "veh_42",
        "type": "passenger",        # one of config.PCU_WEIGHTS' keys
        "direction": "N",           # approach direction, one of config.DIRECTIONS
        "speed": 3.2,               # m/s
        "waiting_time": 5.0,        # accumulated seconds stopped/near-stopped
                                    # (e.g. SUMO's traci.vehicle.getWaitingTime)
        "distance_to_junction": 40.0,   # optional, metres to the stop line
        "lane_id": "N2C_0",              # optional, informational
    }

`waiting_time` and `speed` default to 0.0 if missing so a partially-filled
vehicle record never crashes the calculation; the vehicle is simply treated
as free-flowing / not yet waited.
"""

from typing import Dict, List, Optional

import config

# A vehicle is considered "halted" / part of the queue once its speed drops
# below this threshold (m/s). SUMO's own default halting threshold is
# similar (~0.1 m/s); this is set a little higher to also catch vehicles
# crawling forward in a queue rather than fully stopped.
QUEUE_SPEED_THRESHOLD_MPS = 0.3

# Fallback road length (metres) used for any direction not explicitly
# provided in `road_lengths`. Override with real per-approach lengths
# (e.g. pulled from the SUMO .net.xml edge lengths) for accurate density.
DEFAULT_ROAD_LENGTH_METERS = 100.0

# Average bumper-to-bumper spacing (metres) one PCU-equivalent vehicle
# occupies at jam density: average vehicle length + minimum standstill
# gap. Mirrors the same physical assumption blockage_detection.py already
# uses (its default_vehicle_length_m=5.0 + jam_spacing_m=1.5 = 6.5m) so
# that a "density > 65%" check means the same thing everywhere in the
# pipeline, per the project requirement that density be normalized
# consistently across modules rather than each module inventing its own
# scale.
DEFAULT_JAM_SPACING_PER_PCU_METERS = 6.5


class TrafficMetricCalculator:
    """Computes wait time, PCU-weighted density, and queue length per approach."""

    def __init__(
        self,
        road_lengths: Optional[Dict[str, float]] = None,
        pcu_weights: Optional[Dict[str, float]] = None,
        jam_spacing_per_pcu_m: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        road_lengths : dict, optional
            {"N": 120.0, "E": 95.0, ...} metres of usable approach length
            per direction. Any direction not present falls back to
            DEFAULT_ROAD_LENGTH_METERS.
        pcu_weights : dict, optional
            Overrides config.PCU_WEIGHTS if you need per-instance tuning.
        jam_spacing_per_pcu_m : float, optional
            Bumper-to-bumper spacing (metres) one PCU-equivalent vehicle
            occupies at jam density, used to normalize density into a
            proper 0..1+ occupancy ratio (see module docstring / section
            11: "density percentage must be normalized consistently").
            Defaults to DEFAULT_JAM_SPACING_PER_PCU_METERS.
        """
        self.road_lengths = {
            d: (road_lengths or {}).get(d, DEFAULT_ROAD_LENGTH_METERS)
            for d in config.DIRECTIONS
        }
        self.pcu_weights = pcu_weights or config.PCU_WEIGHTS
        self.jam_spacing_per_pcu_m = (
            jam_spacing_per_pcu_m or DEFAULT_JAM_SPACING_PER_PCU_METERS
        )
        self.last_metrics: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def calculate(
        self,
        vehicle_data: List[Dict],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-approach traffic metrics from raw vehicle data.

        Parameters
        ----------
        vehicle_data : list[dict]
            Raw per-vehicle data for this step, from data_collection.py.

        Returns
        -------
        dict
            Per-approach metrics, one entry for every config.DIRECTIONS
            value (see module docstring for shape).
        """
        by_direction = self._group_by_direction(vehicle_data)

        metrics: Dict[str, Dict[str, float]] = {}
        for direction in config.DIRECTIONS:
            vehicles = by_direction.get(direction, [])

            density = self._calculate_density(vehicles, direction)
            queue_length = self._calculate_queue_length(vehicles)
            wait_time, avg_wait_time = self._calculate_wait_time(vehicles)

            metrics[direction] = {
                "density": density,
                "queue_length": queue_length,
                "wait_time": wait_time,
                "avg_wait_time": avg_wait_time,
                "vehicle_count": len(vehicles),
            }

        # Step 6: store for later analysis / visualisation.
        self.last_metrics = metrics
        return metrics

    # ------------------------------------------------------------------ #
    # Step 1: group raw vehicle data by approach
    # ------------------------------------------------------------------ #
    @staticmethod
    def _group_by_direction(vehicle_data: List[Dict]) -> Dict[str, List[Dict]]:
        by_direction: Dict[str, List[Dict]] = {d: [] for d in config.DIRECTIONS}
        for vehicle in vehicle_data:
            direction = vehicle.get("direction")
            if direction in by_direction:
                by_direction[direction].append(vehicle)
            # Vehicles with an unknown/missing direction are silently
            # excluded from all approach metrics rather than crashing --
            # they simply don't belong to a phase we can serve.
        return by_direction

    # ------------------------------------------------------------------ #
    # Step 3: PCU-weighted density
    # ------------------------------------------------------------------ #
    def _calculate_density(self, vehicles: List[Dict], direction: str) -> float:
        """
        Normalized occupancy ratio: current PCU load divided by the road's
        jam-density PCU capacity (how many PCU-equivalent vehicles could
        physically fit bumper-to-bumper on this approach). 0.0 = empty,
        1.0 = jam-packed; values above 1.0 are possible under
        oversaturation, matching how RQ > 1.0 is treated elsewhere in this
        pipeline (see blockage_detection.py). This satisfies the project
        requirement that "the density percentage must be normalized
        consistently" -- raw PCU-per-metre would otherwise mean a
        different thing on every approach depending on its length.
        """
        total_pcu = sum(
            self.pcu_weights.get(v.get("type", "default"), self.pcu_weights["default"])
            for v in vehicles
        )
        road_length = self.road_lengths.get(direction, DEFAULT_ROAD_LENGTH_METERS)
        if road_length <= 0:
            return 0.0
        max_pcu_capacity = road_length / self.jam_spacing_per_pcu_m
        if max_pcu_capacity <= 0:
            return 0.0
        return total_pcu / max_pcu_capacity

    # ------------------------------------------------------------------ #
    # Step 4: queue length
    # ------------------------------------------------------------------ #
    @staticmethod
    def _calculate_queue_length(vehicles: List[Dict]) -> int:
        """Count of vehicles currently halted (part of the visible queue)."""
        return sum(
            1 for v in vehicles if v.get("speed", 0.0) <= QUEUE_SPEED_THRESHOLD_MPS
        )

    # ------------------------------------------------------------------ #
    # Step 2: wait time
    # ------------------------------------------------------------------ #
    @staticmethod
    def _calculate_wait_time(vehicles: List[Dict]) -> "tuple[float, float]":
        """
        Returns (worst_case_wait, average_wait) across the approach.

        The worst-case (max) wait is what decision_engine's starvation
        override compares against config.MAX_WAIT_SECONDS -- using the
        average here would let a single very-delayed vehicle hide behind
        a lot of fresh arrivals.
        """
        waiting_times = [v.get("waiting_time", 0.0) for v in vehicles]
        if not waiting_times:
            return 0.0, 0.0
        return max(waiting_times), sum(waiting_times) / len(waiting_times)