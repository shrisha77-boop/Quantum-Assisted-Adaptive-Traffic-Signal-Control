import os
import math
import sumolib


# ============================================================
# NETWORK FILE
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

NET_FILE = os.path.join(
    BASE_DIR,
    "scenario",
    "network",
    "bengaluru.net.xml"
)


# ============================================================
# SELECTED CORRIDOR COORDINATES
# These are the actual geographic locations we selected.
# ============================================================

CORRIDOR = {
    "J1": (12.923699, 77.617937),
    "J2": (12.924644, 77.618261),
    "J3": (12.927525, 77.620911),
    "J4": (12.925747, 77.625173),
    "J5": (12.924629, 77.628337),
}


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(x1, y1, x2, y2):
    """
    Euclidean distance in SUMO coordinate system.
    """

    return math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )


# ============================================================
# LOAD NETWORK
# ============================================================

print()
print("=" * 110)
print("FINDING ACTUAL TRAFFIC-LIGHT CONTROLLERS")
print("=" * 110)

print()
print("Network file:")
print(NET_FILE)

print()
print("Loading SUMO network...")

net = sumolib.net.readNet(NET_FILE)

print("Network loaded successfully.")


# ============================================================
# GET ALL TRAFFIC LIGHTS
# ============================================================

tls_list = net.getTrafficLights()

print()
print("=" * 110)
print("TRAFFIC-LIGHT CONTROLLERS FOUND")
print("=" * 110)

print()
print(f"Total TLS controllers in network: {len(tls_list)}")


# ============================================================
# BUILD TLS LOCATION LIST
# ============================================================

tls_data = []

for tls in tls_list:

    tls_id = tls.getID()

    # --------------------------------------------------------
    # A TLS may control multiple connections.
    # Get controlled links.
    # --------------------------------------------------------

    controlled_links = tls.getConnections()

    # Some SUMO versions expose TLS coordinate indirectly
    # through controlled junctions.
    junctions = set()

    for connection in controlled_links:

        try:
            from_lane = connection[0]
            to_lane = connection[1]

            from_edge = from_lane.getEdge()
            to_edge = to_lane.getEdge()

            from_node = from_edge.getToNode()
            to_node = to_edge.getFromNode()

            junctions.add(from_node)
            junctions.add(to_node)

        except Exception:
            continue

    # --------------------------------------------------------
    # Calculate approximate TLS position from controlled
    # junction positions.
    # --------------------------------------------------------

    if junctions:

        xs = []
        ys = []

        for node in junctions:

            x, y = node.getCoord()

            xs.append(x)
            ys.append(y)

        tls_x = sum(xs) / len(xs)
        tls_y = sum(ys) / len(ys)

    else:

        tls_x = None
        tls_y = None

    tls_data.append(
        {
            "id": tls_id,
            "x": tls_x,
            "y": tls_y,
            "junctions": junctions,
            "connections": controlled_links,
        }
    )


# ============================================================
# FIND NEAREST TLS FOR EACH CORRIDOR JUNCTION
# ============================================================

print()
print("=" * 110)
print("NEAREST TLS FOR EACH SELECTED CORRIDOR JUNCTION")
print("=" * 110)


nearest_results = {}


for junction_name, (target_lat, target_lon) in CORRIDOR.items():

    print()
    print("-" * 110)
    print(junction_name)
    print("-" * 110)

    # --------------------------------------------------------
    # Convert geographic coordinates to SUMO coordinates.
    # --------------------------------------------------------

    target_x, target_y = net.convertLonLat2XY(
        target_lon,
        target_lat
    )

    print(f"Target latitude  : {target_lat}")
    print(f"Target longitude : {target_lon}")

    print()
    print(f"Target SUMO X    : {target_x:.2f}")
    print(f"Target SUMO Y    : {target_y:.2f}")

    # --------------------------------------------------------
    # Find closest TLS.
    # --------------------------------------------------------

    candidates = []

    for tls in tls_data:

        if tls["x"] is None:
            continue

        d = distance(
            target_x,
            target_y,
            tls["x"],
            tls["y"]
        )

        candidates.append(
            (
                d,
                tls
            )
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    # --------------------------------------------------------
    # Print top 5 closest TLS.
    # --------------------------------------------------------

    print()
    print("Closest traffic-light controllers:")

    for rank, (d, tls) in enumerate(
        candidates[:5],
        start=1
    ):

        print()
        print(f"{rank}. TLS ID : {tls['id']}")
        print(
            f"   Distance : {d:.2f} m"
        )
        print(
            f"   Position : "
            f"({tls['x']:.2f}, {tls['y']:.2f})"
        )
        print(
            f"   Junctions controlled/associated: "
            f"{len(tls['junctions'])}"
        )

    # --------------------------------------------------------
    # Store closest result.
    # --------------------------------------------------------

    if candidates:

        nearest_results[junction_name] = candidates[0]

    else:

        nearest_results[junction_name] = None


# ============================================================
# DETAILED TLS INSPECTION
# ============================================================

print()
print("=" * 110)
print("DETAILED CORRIDOR TLS INFORMATION")
print("=" * 110)


for junction_name, result in nearest_results.items():

    print()
    print("-" * 110)
    print(junction_name)
    print("-" * 110)

    if result is None:

        print("No TLS found.")

        continue

    distance_value, tls = result

    print()
    print(f"Nearest TLS ID : {tls['id']}")
    print(f"Distance       : {distance_value:.2f} m")

    print()
    print(
        f"TLS position   : "
        f"({tls['x']:.2f}, {tls['y']:.2f})"
    )

    # --------------------------------------------------------
    # Associated junctions
    # --------------------------------------------------------

    print()
    print(
        f"Associated junctions: "
        f"{len(tls['junctions'])}"
    )

    for node in tls["junctions"]:

        print(
            f"  - {node.getID()}"
        )

    # --------------------------------------------------------
    # Controlled connections
    # --------------------------------------------------------

    print()
    print(
        f"Controlled connections: "
        f"{len(tls['connections'])}"
    )

    for connection in tls["connections"]:

        try:

            from_lane = connection[0]
            to_lane = connection[1]

            from_edge = from_lane.getEdge()
            to_edge = to_lane.getEdge()

            print(
                f"  {from_lane.getID()} "
                f"-> "
                f"{to_lane.getID()}"
            )

        except Exception:

            print("  [Unable to decode connection]")


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 110)
print("FINAL CORRIDOR TLS SUMMARY")
print("=" * 110)

print()

for junction_name, result in nearest_results.items():

    if result is None:

        print(
            f"{junction_name}: "
            f"NO TLS FOUND"
        )

        continue

    distance_value, tls = result

    print(
        f"{junction_name}: "
        f"TLS = {tls['id']} "
        f"| Distance = {distance_value:.2f} m"
    )


print()
print("=" * 110)
print("TLS SEARCH COMPLETE")
print("=" * 110)
print()