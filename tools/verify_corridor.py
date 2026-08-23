import os
import sys
import math
import heapq
import sumolib


NET_FILE = "scenario/network/bengaluru.net.xml"

TARGETS = [
    ("J1", 12.923699, 77.617937),
    ("J2", 12.924644, 77.618261),
    ("J3", 12.927525, 77.620911),
    ("J4", 12.925747, 77.625173),
    ("J5", 12.924629, 77.628337),
]


# ============================================================
# LOAD NETWORK
# ============================================================

print("\nLoading SUMO network...")

if not os.path.exists(NET_FILE):
    print("ERROR: Network file not found:")
    print(NET_FILE)
    sys.exit(1)

net = sumolib.net.readNet(NET_FILE)

print("Network loaded successfully.\n")


# ============================================================
# FIND ACTUAL TRAFFIC LIGHT JUNCTIONS
# ============================================================

traffic_lights = [
    node
    for node in net.getNodes()
    if node.getType() == "traffic_light"
]


def distance(x1, y1, x2, y2):
    return math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )


selected = []


# ============================================================
# MATCH TARGET COORDINATES TO ACTUAL SUMO NODES
# ============================================================

print("=" * 100)
print("SELECTED CORRIDOR JUNCTIONS")
print("=" * 100)

for name, lat, lon in TARGETS:

    x, y = net.convertLonLat2XY(lon, lat)

    nearest = min(
        traffic_lights,
        key=lambda node: distance(
            node.getCoord()[0],
            node.getCoord()[1],
            x,
            y
        )
    )

    selected.append(nearest)

    print(
        f"{name}: {nearest.getID()} | "
        f"Distance = "
        f"{distance(nearest.getCoord()[0], nearest.getCoord()[1], x, y):.2f} m"
    )


# ============================================================
# DIJKSTRA SHORTEST PATH
# ============================================================

def shortest_path(source, target):

    queue = [(0.0, source.getID())]

    distances = {
        source.getID(): 0.0
    }

    previous = {}

    node_map = {
        node.getID(): node
        for node in net.getNodes()
    }

    while queue:

        current_distance, current_id = heapq.heappop(queue)

        if current_id == target.getID():
            break

        if current_distance > distances.get(current_id, float("inf")):
            continue

        current_node = node_map[current_id]

        for edge in current_node.getOutgoing():

            next_node = edge.getToNode()
            next_id = next_node.getID()

            weight = edge.getLength()

            new_distance = current_distance + weight

            if new_distance < distances.get(
                next_id,
                float("inf")
            ):

                distances[next_id] = new_distance

                previous[next_id] = (
                    current_id,
                    edge
                )

                heapq.heappush(
                    queue,
                    (new_distance, next_id)
                )

    if target.getID() not in distances:
        return None

    # Reconstruct path

    path_edges = []

    current = target.getID()

    while current != source.getID():

        previous_node, edge = previous[current]

        path_edges.append(edge)

        current = previous_node

    path_edges.reverse()

    return path_edges, distances[target.getID()]


# ============================================================
# CHECK ALL CONSECUTIVE JUNCTIONS
# ============================================================

print("\n")
print("=" * 100)
print("ACTUAL NETWORK PATHS")
print("=" * 100)


all_corridor_edges = []


for i in range(len(selected) - 1):

    source = selected[i]
    target = selected[i + 1]

    print("\n" + "-" * 100)

    print(
        f"J{i + 1} → J{i + 2}"
    )

    print(f"Source: {source.getID()}")
    print(f"Target: {target.getID()}")

    result = shortest_path(source, target)

    if result is None:

        print("\nPATH: NOT FOUND")

    else:

        path_edges, total_distance = result

        print("\nPATH: FOUND")

        print(
            f"Network distance: "
            f"{total_distance:.2f} m "
            f"({total_distance / 1000:.3f} km)"
        )

        print(
            f"Number of edges: "
            f"{len(path_edges)}"
        )

        print("\nEdges:")

        for edge in path_edges:

            print(
                f"  {edge.getID()} "
                f"({edge.getLength():.2f} m) "
                f"→ {edge.getToNode().getID()}"
            )

        all_corridor_edges.extend(path_edges)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 100)
print("FINAL CORRIDOR VERIFICATION")
print("=" * 100)

print("\nCorridor:")

for i, node in enumerate(selected, start=1):

    print(
        f"J{i} = {node.getID()}"
    )


print("\n")

print(
    "If all four paths above say 'PATH: FOUND', "
    "the five junctions are connected through the SUMO network."
)

print(
    "\nNext we will identify the actual traffic-light "
    "controllers, incoming lanes and corridor movement edges."
)

print("\n")
print("=" * 100)
print("VERIFICATION COMPLETE")
print("=" * 100)