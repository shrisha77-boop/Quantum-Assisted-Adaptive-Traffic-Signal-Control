"""
Static traffic signal configuration for the Bengaluru corridor.

This module contains only scenario configuration.

It does not:
- connect to SUMO
- use TraCI
- collect traffic data
- calculate traffic metrics
- optimize signal phases
- build QUBOs
- control traffic lights
"""


JUNCTIONS = {
    "J1": {
        "tls_id": "1607769741",
        "candidate_roads": [
            "-254053176#2",
            "147575119#1",
            "-147575122",
            "254053176#1",
            "383732625#3",
        ],
        "green_phases": [0, 2, 4],
    },

    "J2": {
        "tls_id": "cluster_10043935988_10043935989_1070361799",
        "candidate_roads": [
            "111814614#6",
            "133763768#0",
            "1223486358",
        ],
        "green_phases": [0, 2, 4],
    },

    "J3": {
        "tls_id": "cluster_1070361724_11307884034_11882719580_11882719581_#9more",
        "candidate_roads": [
            "1110178545",
            "24231604#4",
            "133763768#6",
            "1290479888#2",
        ],
        "green_phases": [0, 2, 4, 6],
    },

    "J4": {
        "tls_id": "cluster_13058420509_2571434008_308915752_6137597661",
        "candidate_roads": [
            "1411121774#3",
            "-269436416#0",
            "150510724#4",
        ],
        "green_phases": [0, 2],
    },

    "J5": {
        "tls_id": "cluster_12074449281_1759977210",
        "candidate_roads": [
            "1159872650#18",
            "457214835#12",
            "-1311812958",
            "631197284#3",
        ],
        "green_phases": [0, 2, 4],
    },
}


def get_junction_config(junction_id):
    """Return configuration for a junction."""
    if junction_id not in JUNCTIONS:
        raise ValueError(
            f"Unknown junction ID: {junction_id}"
        )

    return JUNCTIONS[junction_id]


def get_all_junctions():
    """Return all configured junction IDs."""
    return list(JUNCTIONS.keys())