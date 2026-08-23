import os
import sumolib
from collections import deque


# ============================================================
# NETWORK
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NET_FILE = os.path.join(
    BASE_DIR,
    "scenario",
    "network",
    "bengaluru.net.xml"
)


# ============================================================
# CORRIDOR JUNCTIONS
# ============================================================

J5 = "cluster_12074449281_1759977210"
J4 = "cluster_13058420509_2571434008_308915752_6137597661"
J3 = "cluster_1070361724_11307884034_11882719580_11882719581_#9more"
J2 = "cluster_10043935988_10043935989_1070361799"
J1 = "1607769741"


CORRIDOR = [
    ("J5", J5, "J4", J4),
    ("J4", J4, "J3", J3),
    ("J3", J3, "J2", J2),
    ("J2", J2, "J1", J1),
]


# ============================================================
# LOAD NETWORK
# ============================================================

print()
print("=" * 110)
print("REVERSE CORRIDOR INSPECTION")
print("=" * 110)

print()
print("Network file:")
print(NET_FILE)

print()
print("Loading SUMO network...")

net = sumolib.net.readNet(NET_FILE)

print("Network loaded successfully.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_edges_leaving_node(node):
    """
    Return all normal outgoing edges from a SUMO node.
    Internal junction edges are ignored.
    """

    result = []

    for edge in node.getOutgoing():

        edge_id = edge.getID()

        # Ignore SUMO internal edges
        if edge_id.startswith(":"):
            continue

        result.append(edge)

    return result


def get_edges_entering_node(node):
    """
    Return all normal incoming edges to a SUMO node.
    Internal junction edges are ignored.
    """

    result = []

    for edge in node.getIncoming():

        edge_id = edge.getID()

        if edge_id.startswith(":"):
            continue

        result.append(edge)

    return result


def find_edge_path(source_node, target_node):
    """
    Breadth-first search through SUMO edges.

    We use the actual edge graph instead of
    sumolib.net.getShortestPath(), because the installed
    SUMO version has compatibility issues with that method.
    """

    queue = deque()

    visited = set()

    parent = {}

    # Start from every edge leaving source junction
    start_edges = get_edges_leaving_node(source_node)

    for edge in start_edges:

        edge_id = edge.getID()

        queue.append(edge)
        visited.add(edge_id)

        parent[edge_id] = None

    while queue:

        current_edge = queue.popleft()

        current_id = current_edge.getID()

        # ----------------------------------------------------
        # Have we reached target junction?
        # ----------------------------------------------------

        if current_edge.getToNode() == target_node:

            path = []

            edge = current_edge

            while edge is not None:

                path.append(edge)

                previous_id = parent[edge.getID()]

                if previous_id is None:
                    break

                edge = net.getEdge(previous_id)

            path.reverse()

            return path

        # ----------------------------------------------------
        # Continue through outgoing edges
        # ----------------------------------------------------

        next_node = current_edge.getToNode()

        for next_edge in get_edges_leaving_node(next_node):

            next_id = next_edge.getID()

            if next_id in visited:
                continue

            visited.add(next_id)

            parent[next_id] = current_id

            queue.append(next_edge)

    return None


# ============================================================
# FIND REVERSE PATHS
# ============================================================

print()
print("=" * 110)
print("REVERSE-DIRECTION CORRIDOR PATHS")
print("=" * 110)


reverse_paths = []


for source_name, source_id, target_name, target_id in CORRIDOR:

    print()
    print("-" * 110)
    print(f"{source_name} -> {target_name}")
    print("-" * 110)

    try:

        source = net.getNode(source_id)
        target = net.getNode(target_id)

    except KeyError as error:

        print()
        print("ERROR: Junction not found.")
        print(f"Missing ID: {error}")

        reverse_paths.append({
            "source": source_name,
            "target": target_name,
            "edges": []
        })

        continue

    path = find_edge_path(source, target)

    if path is None:

        print()
        print("PATH: NOT FOUND")

        reverse_paths.append({
            "source": source_name,
            "target": target_name,
            "edges": []
        })

        continue

    # --------------------------------------------------------
    # Calculate distance
    # --------------------------------------------------------

    distance = sum(
        edge.getLength()
        for edge in path
    )

    print()
    print("PATH: FOUND")

    print(
        f"Network distance: "
        f"{distance:.2f} m "
        f"({distance / 1000:.3f} km)"
    )

    print(
        f"Number of edges: {len(path)}"
    )

    print()
    print("Edges:")

    for index, edge in enumerate(path, start=1):

        print()
        print(f"[{index}]")

        print(
            f"Edge ID : {edge.getID()}"
        )

        print(
            f"Length  : {edge.getLength():.2f} m"
        )

        print(
            f"Speed   : {edge.getSpeed():.2f} m/s"
        )

        print(
            f"Lanes   : {len(edge.getLanes())}"
        )

        for lane in edge.getLanes():

            print(
                f"    Lane ID : {lane.getID()} "
                f"| Length = {lane.getLength():.2f} m"
            )

    reverse_paths.append({
        "source": source_name,
        "target": target_name,
        "edges": path,
        "distance": distance
    })


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 110)
print("REVERSE CORRIDOR SUMMARY")
print("=" * 110)


successful = 0


for item in reverse_paths:

    print()
    print(
        f"{item['source']} -> {item['target']}"
    )

    edges = item["edges"]

    if not edges:

        print("  PATH NOT FOUND")

        continue

    successful += 1

    distance = item["distance"]

    print(
        f"  Number of edges : {len(edges)}"
    )

    print(
        f"  Total distance  : "
        f"{distance:.2f} m "
        f"({distance / 1000:.3f} km)"
    )

    print("  Edge sequence:")

    for edge in edges:

        print(
            f"    {edge.getID()} "
            f"({edge.getLength():.2f} m)"
        )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 110)
print("REVERSE CORRIDOR STATUS")
print("=" * 110)

print()
print(
    f"Successfully found: "
    f"{successful}/4 reverse paths"
)

if successful == 4:

    print()
    print("SUCCESS")
    print("All reverse corridor paths were found.")

else:

    print()
    print("WARNING")
    print(
        "Some reverse paths could not be found."
    )


print()
print("=" * 110)
print("REVERSE CORRIDOR INSPECTION COMPLETE")
print("=" * 110)
print()