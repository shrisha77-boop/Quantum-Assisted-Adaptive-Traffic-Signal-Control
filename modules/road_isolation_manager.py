# 1. Receive blocked road information.
# 2. Apply an isolation flag to blocked roads.
# 3. Prevent the Decision Engine from selecting isolated roads.
# 4. Notify neighbouring junctions
# 5. Continue checking the road status.
# 6. Remove the isolation flag when the blockage is cleared.
# 7. Restore the road to normal operation.
# 8. Send the updated road availability to the Decision Engine.
"""
road_isolation_manager.py
================================================================================
Road Isolation Manager
--------------------------------------------------------------------------------
Position in the pipeline:

    SUMO -> TraCI API -> traci_interface.py -> data_collection.py
         -> blockage_detection.py -> road_isolation_manager.py (THIS MODULE)
         -> decision_engine.py -> signal_controller.py

This module MUST NEVER communicate directly with TraCI.
This module MUST NEVER receive data directly from the Data Collection Layer.
It ONLY consumes blockage reports already produced by blockage_detection.py
and trusts their reported status as-is.

This module is responsible ONLY for:
    - Maintaining a live Road Availability Table (isolated / available roads).
    - Applying / removing isolation flags strictly on confirmed status changes.
    - Broadcasting ROAD_ISOLATED / ROAD_RESTORED messages to adjacent junctions.
    - Tracking (but never acting on) neighbouring junctions' road status.
    - Preparing the Candidate Road List for the Decision Engine.

This module must NEVER:
    - Recalculate or infer blockage conditions.
    - Compute traffic metrics.
    - Perform optimisation.
    - Control traffic signals.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class RoadIsolationError(Exception):
    """
    Raised when a blockage report is malformed in a way that section 39
    ("Invalid Conditions") requires this system to fail clearly on,
    rather than silently skip and leave the Road Availability Table
    stale/incomplete.
    """


# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

@dataclass
class IsolationConfig:
    """Centralized, overridable configuration for road isolation handling."""

    # junction_id -> list of adjacent/connected junction ids to notify.
    # field(default_factory=dict) gives EVERY IsolationConfig instance its
    # own independent dict. A plain class-level `= {}` default would be a
    # single dict object SHARED by every instance that doesn't explicitly
    # reassign it -- in a multi-junction corridor, an in-place update
    # (config_a.NEIGHBOR_JUNCTIONS['J1'] = [...]) would silently become
    # visible through every other junction's config too.
    NEIGHBOR_JUNCTIONS: Dict[str, List[str]] = field(default_factory=dict)


class RoadStatus(str, Enum):
    """Status values trusted verbatim from blockage_detection.py."""
    BLOCKED = "BLOCKED"
    CLEAR = "CLEAR"


class MessageType(str, Enum):
    ROAD_ISOLATED = "ROAD_ISOLATED"
    ROAD_RESTORED = "ROAD_RESTORED"


# ============================================================================
# SECTION 2: DATA STRUCTURES
# ============================================================================

@dataclass
class RoadRecord:
    """One row of the Road Availability Table."""
    road_id: str
    junction_id: str
    status: str                 # "BLOCKED" or "CLEAR", trusted from the report
    isolated: bool
    available: bool
    last_updated: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IsolationMessage:
    """
    Minimal inter-junction coordination message. Intentionally excludes
    traffic statistics (queue ratio, waiting time, speed ratio, reasons) --
    only what neighbours need for awareness/coordination.
    """
    message_type: str            # MessageType.ROAD_ISOLATED / ROAD_RESTORED
    road_id: str
    junction_id: str
    timestamp: float
    target_junction: str = ""    # filled in when addressed to a specific neighbour

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# SECTION 3: MAIN CLASS
# ============================================================================

class RoadIsolationManager:
    """
    Maintains road isolation state for a single junction based purely on
    trusted blockage reports, and coordinates awareness (not control) with
    adjacent junctions' RoadIsolationManager instances via messages.
    """

    def __init__(
        self,
        junction_id: str,
        config: Optional[IsolationConfig] = None,
        neighbor_junctions: Optional[Sequence[str]] = None,
    ) -> None:
        self.junction_id = junction_id
        self.config = config or IsolationConfig()
        self.neighbor_junctions: List[str] = list(
            neighbor_junctions
            if neighbor_junctions is not None
            else self.config.NEIGHBOR_JUNCTIONS.get(junction_id, [])
        )

        # STEP 3: live Road Availability Table (road_id -> RoadRecord).
        self.road_availability: Dict[str, RoadRecord] = {}

        # STEP 6: separate table for neighbour-reported road status.
        # Local availability is NEVER modified from this table.
        self.neighbour_road_status: Dict[str, Dict[str, Any]] = {}

        # Outbox drained by the orchestrator after each processing pass.
        self._outgoing_messages: List[IsolationMessage] = []

        self._last_sim_time: float = 0.0

    # ------------------------------------------------------------------
    # PUBLIC ENTRY POINTS
    # ------------------------------------------------------------------
    def process_report(self, report: Dict[str, Any]) -> None:
        """
        STEP 1-2, 4, 7-9: Ingest a single blockage report from
        blockage_detection.py, update the Road Availability Table, and queue
        an isolation/restoration broadcast ONLY on an actual status change.
        """
        road_id = str(report.get("lane_id", ""))
        junction_id = str(report.get("junction_id", self.junction_id))
        status_raw = str(report.get("status", "")).upper()
        timestamp = float(report.get("timestamp", self._last_sim_time))
        self._last_sim_time = timestamp

        if not road_id:
            raise RoadIsolationError(
                f"Blockage report is missing 'lane_id': {report!r}"
            )

        if status_raw not in (RoadStatus.BLOCKED.value, RoadStatus.CLEAR.value):
            raise RoadIsolationError(
                f"Blockage report for road {road_id!r} has an unrecognised "
                f"status {status_raw!r} (expected 'BLOCKED' or 'CLEAR'): {report!r}"
            )

        existing = self.road_availability.get(road_id)
        previous_status = existing.status if existing is not None else None

        # STEP 4/7: no duplicate updates when status has not actually changed.
        if previous_status == status_raw:
            return

        # STEP 2 / STEP 8: apply or remove the isolation flag, trusting the
        # reported status as-is (no recalculation).
        isolated = status_raw == RoadStatus.BLOCKED.value
        available = not isolated

        self.road_availability[road_id] = RoadRecord(
            road_id=road_id,
            junction_id=junction_id,
            status=status_raw,
            isolated=isolated,
            available=available,
            last_updated=timestamp,
        )

        # STEP 5 / STEP 9: broadcast only on an actual CLEAR<->BLOCKED transition.
        if previous_status == RoadStatus.CLEAR.value and status_raw == RoadStatus.BLOCKED.value:
            self._queue_broadcast(MessageType.ROAD_ISOLATED, road_id, junction_id, timestamp)
        elif previous_status == RoadStatus.BLOCKED.value and status_raw == RoadStatus.CLEAR.value:
            self._queue_broadcast(MessageType.ROAD_RESTORED, road_id, junction_id, timestamp)
        elif previous_status is None and status_raw == RoadStatus.BLOCKED.value:
            # First-ever report for this road arrives already BLOCKED.
            self._queue_broadcast(MessageType.ROAD_ISOLATED, road_id, junction_id, timestamp)

    def process_reports(self, reports: Sequence[Dict[str, Any]]) -> None:
        """Convenience batch entry point for multiple blockage reports in one step."""
        for report in reports:
            self.process_report(report)

    # ------------------------------------------------------------------
    # STEP 5 / STEP 9: BROADCASTING
    # ------------------------------------------------------------------
    def _queue_broadcast(
        self, message_type: MessageType, road_id: str, junction_id: str, timestamp: float
    ) -> None:
        """Queue a minimal isolation/restoration message for every configured neighbour."""
        for neighbor_id in self.neighbor_junctions:
            self._outgoing_messages.append(
                IsolationMessage(
                    message_type=message_type.value,
                    road_id=road_id,
                    junction_id=junction_id,
                    timestamp=timestamp,
                    target_junction=neighbor_id,
                )
            )

    def get_pending_broadcasts(self) -> List[IsolationMessage]:
        """Drain and return outbound messages for the orchestrator to deliver
        to the relevant neighbour RoadIsolationManager instances (via their
        receive_neighbor_message() method)."""
        messages = self._outgoing_messages
        self._outgoing_messages = []
        return messages

    # ------------------------------------------------------------------
    # STEP 6: RECEIVING NEIGHBOUR UPDATES
    # ------------------------------------------------------------------
    def receive_neighbor_message(self, message: IsolationMessage) -> None:
        """
        Store neighbour-reported isolation status for awareness/coordination
        ONLY. Local road_availability is never touched here.
        """
        self.neighbour_road_status[message.road_id] = {
            "junction_id": message.junction_id,
            "isolated": message.message_type == MessageType.ROAD_ISOLATED.value,
            "message_type": message.message_type,
            "last_updated": message.timestamp,
        }

    # ------------------------------------------------------------------
    # STEP 10/11: CANDIDATE ROAD LIST + OUTPUT
    # ------------------------------------------------------------------
    def _build_candidate_roads(self) -> List[str]:
        """
        STEP 10: candidate roads are every road that is BOTH available and
        not isolated. No optimisation happens here -- just list preparation.
        """
        return [
            record.road_id
            for record in self.road_availability.values()
            if record.available and not record.isolated
        ]

    def _build_isolated_roads(self) -> List[str]:
        return [
            record.road_id
            for record in self.road_availability.values()
            if record.isolated
        ]

    def _build_available_roads(self) -> List[str]:
        return [
            record.road_id
            for record in self.road_availability.values()
            if record.available
        ]

    def get_output(self) -> Dict[str, Any]:
        """
        STEP 11: assemble the structured output consumed by the Decision
        Engine. The Decision Engine must use ONLY candidate_roads when
        constructing the optimisation problem and must ignore every isolated
        road entirely.
        """
        return {
            "road_availability": {
                road_id: record.to_dict()
                for road_id, record in self.road_availability.items()
            },
            "candidate_roads": self._build_candidate_roads(),
            "isolated_roads": self._build_isolated_roads(),
            "available_roads": self._build_available_roads(),
            "neighbour_road_status": dict(self.neighbour_road_status),
            "last_updated": self._last_sim_time,
        }


# ============================================================================
# SECTION 4: LIGHTWEIGHT DEMONSTRATION (no TraCI / SUMO involved)
# ============================================================================
if __name__ == "__main__":
    # Synthetic demo using report shapes identical to what blockage_detection.py
    # would produce. No simulation or TraCI calls occur here.

    config = IsolationConfig()
    config.NEIGHBOR_JUNCTIONS = {"J1": ["J2", "J3"]}

    manager = RoadIsolationManager(junction_id="J1", config=config)

    # Road becomes blocked.
    manager.process_report(
        {
            "lane_id": "edge_in_north_0",
            "junction_id": "J1",
            "status": "BLOCKED",
            "queue_storage_ratio": 0.95,
            "average_waiting_time": 120.0,
            "speed_ratio": 0.05,
            "reasons": ["queue_overflow"],
            "timestamp": 100.0,
        }
    )
    print("After BLOCKED:", manager.get_output())
    print("Broadcasts:", [m.to_dict() for m in manager.get_pending_broadcasts()])

    # Duplicate report with same status -- should NOT re-broadcast.
    manager.process_report(
        {
            "lane_id": "edge_in_north_0",
            "junction_id": "J1",
            "status": "BLOCKED",
            "timestamp": 110.0,
        }
    )
    print("Broadcasts after duplicate BLOCKED report:", manager.get_pending_broadcasts())

    # Road clears.
    manager.process_report(
        {
            "lane_id": "edge_in_north_0",
            "junction_id": "J1",
            "status": "CLEAR",
            "timestamp": 140.0,
        }
    )
    print("After CLEAR:", manager.get_output())
    print("Broadcasts:", [m.to_dict() for m in manager.get_pending_broadcasts()])