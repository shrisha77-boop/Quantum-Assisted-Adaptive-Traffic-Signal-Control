"""
Scenario configuration for the Bengaluru adaptive traffic signal corridor.

This module contains only static information about the SUMO scenario:
- Junction IDs
- SUMO traffic-light IDs
- Corridor order
- Signal phase information
- Incoming roads relevant to each junction

It does not:
- connect to SUMO
- collect traffic data
- calculate traffic metrics
- build QUBOs
- solve optimization problems
- control traffic signals
"""

from typing import Dict, List, Any


# ============================================================
# SUMO SCENARIO
# ============================================================

SCENARIO_NAME = "Bengaluru Adaptive Traffic Signal Corridor"

SUMO_CONFIG = "scenario/config/corridor.sumocfg"

SIMULATION_BEGIN = 0
SIMULATION_END = 3600
STEP_LENGTH = 1.0


# ============================================================
# CORRIDOR
# ============================================================

JUNCTION_ORDER: List[str] = [
    "J1",
    "J2",
    "J3",
    "J4",
    "J5",
]


# ============================================================
# TRAFFIC LIGHT CONFIGURATION
# ============================================================

JUNCTIONS: Dict[str, Dict[str, Any]] = {

    "J1": {
        "tls_id": "1607769741",

        "incoming_edges": [
            "-254053176#2",
            "147575119#1",
            "-147575122",
            "254053176#1",
            "383732625#3",
        ],

        "phase_durations": [
            27.0,
            3.0,
            27.0,
            3.0,
            27.0,
            3.0,
        ],

        "phase_states": [
            "GGgggrrrrrrrrrrGGGggrrrrr",
            "yyyyyrrrrrrrrrryyyyyrrrrr",
            "rrrrrGGgggrrrrrrrrrrGGGgg",
            "rrrrryyyyyrrrrrrrrrryyyyy",
            "rrrrrrrrrrGGGGgGrrrrrrrrr",
            "rrrrrrrrrryyyyyGrrrrrrrrr",
        ],
    },

    "J2": {
        "tls_id": "cluster_10043935988_10043935989_1070361799",

        "incoming_edges": [
            "111814614#6",
            "133763768#0",
            "1223486358",
        ],

        "phase_durations": [
            33.0,
            6.0,
            6.0,
            6.0,
            33.0,
            6.0,
        ],

        "phase_states": [
            "GGrgGGgrrr",
            "yyrgyyyrrr",
            "rrgGrrrrrr",
            "rryyrrrrrr",
            "rrrrrrrGGG",
            "rrrrrrryyy",
        ],
    },

    "J3": {
        "tls_id": (
            "cluster_1070361724_11307884034_"
            "11882719580_11882719581_#9more"
        ),

        "incoming_edges": [
            "1110178545",
            "24231604#4",
            "133763768#6",
            "1290479888#2",
        ],

        "phase_durations": [
            27.0,
            6.0,
            6.0,
            6.0,
            27.0,
            6.0,
            6.0,
            6.0,
        ],

        "phase_states": [
            "rrrrrrGGGrrrrrrrrGGGrr",
            "rrrrrryyyrrrrrrrryyyrr",
            "rrrrrGrrrgGrrrrrGrrrgG",
            "rrrrryrrryyrrrrryrrryy",
            "rGGrrrrrrrrgGGggrrrrrr",
            "ryyrrrrrrrrgyyggrrrrrr",
            "GrrgGrrrrrrGrrgGrrrrrr",
            "yrryyrrrrrryrryyrrrrrr",
        ],
    },

    "J4": {
        "tls_id": (
            "cluster_13058420509_2571434008_"
            "308915752_6137597661"
        ),

        "incoming_edges": [
            "1411121774#3",
            "-269436416#0",
            "150510724#4",
        ],

        "phase_durations": [
            39.0,
            6.0,
            39.0,
            6.0,
        ],

        "phase_states": [
            "GGggrrrGGGGg",
            "yyyyrrrGyyyy",
            "rrrrGGgGrrrr",
            "rrrryyyGrrrr",
        ],
    },

    "J5": {
        "tls_id": "cluster_12074449281_1759977210",

        "incoming_edges": [
            "1159872650#18",
            "457214835#12",
            "-1311812958",
            "631197284#3",
        ],

        "phase_durations": [
            25.0,
            5.0,
            25.0,
            5.0,
            25.0,
            5.0,
        ],

        "phase_states": [
            "GGgrrrGggrrr",
            "yyyrrryyyrrr",
            "rrrGGGrrrrrr",
            "rrryyyrrrrrr",
            "GrrrrrrrrGGg",
            "Grrrrrrrryyy",
        ],
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_junction(junction_id: str) -> Dict[str, Any]:
    """Return configuration for a junction."""
    if junction_id not in JUNCTIONS:
        raise KeyError(
            f"Unknown junction ID: {junction_id}. "
            f"Available junctions: {list(JUNCTIONS)}"
        )

    return JUNCTIONS[junction_id]


def get_tls_id(junction_id: str) -> str:
    """Return the SUMO TLS ID for a junction."""
    return get_junction(junction_id)["tls_id"]


def get_incoming_edges(junction_id: str) -> List[str]:
    """Return incoming edges associated with a junction."""
    return list(get_junction(junction_id)["incoming_edges"])


def get_phase_durations(junction_id: str) -> List[float]:
    """Return signal phase durations."""
    return list(get_junction(junction_id)["phase_durations"])


def get_phase_states(junction_id: str) -> List[str]:
    """Return signal phase states."""
    return list(get_junction(junction_id)["phase_states"])


def validate_configuration() -> None:
    """Validate the static scenario configuration."""

    if not JUNCTION_ORDER:
        raise ValueError("No junctions are configured.")

    for junction_id in JUNCTION_ORDER:

        if junction_id not in JUNCTIONS:
            raise ValueError(
                f"Junction {junction_id} is missing from JUNCTIONS."
            )

        junction = JUNCTIONS[junction_id]

        if not junction["tls_id"]:
            raise ValueError(
                f"{junction_id} has no TLS ID."
            )

        if not junction["incoming_edges"]:
            raise ValueError(
                f"{junction_id} has no incoming edges."
            )

        if not junction["phase_durations"]:
            raise ValueError(
                f"{junction_id} has no phase durations."
            )

        if not junction["phase_states"]:
            raise ValueError(
                f"{junction_id} has no phase states."
            )

        if len(junction["phase_durations"]) != len(
            junction["phase_states"]
        ):
            raise ValueError(
                f"{junction_id}: phase duration/state count mismatch."
            )


if __name__ == "__main__":

    validate_configuration()

    print("=" * 70)
    print("SCENARIO CONFIGURATION")
    print("=" * 70)

    print(f"Scenario : {SCENARIO_NAME}")
    print(f"Junctions: {', '.join(JUNCTION_ORDER)}")

    print()

    for junction_id in JUNCTION_ORDER:

        junction = get_junction(junction_id)

        print(f"{junction_id}")
        print(f"  TLS ID          : {junction['tls_id']}")
        print(
            f"  Incoming edges  : "
            f"{len(junction['incoming_edges'])}"
        )
        print(
            f"  Signal phases   : "
            f"{len(junction['phase_durations'])}"
        )

    print()
    print("Configuration validation: PASSED")
    print("=" * 70)