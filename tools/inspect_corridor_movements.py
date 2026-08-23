import os
import sumolib


# ============================================================
# NETWORK FILE
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NET_FILE = os.path.join(
    BASE_DIR,
    "scenario",
    "network",
    "bengaluru.net.xml"
)


# ============================================================
# SELECTED CORRIDOR JUNCTIONS
# ============================================================

CORRIDOR = [
    "1607769741",
    "cluster_10043935988_10043935989_1070361799",
    "cluster_1070361724_11307884034_11882719580_11882719581_#9more",
    "cluster_13058420509_2571434008_308915752_6137597661",
    "cluster_12074449281_1759977210",
]


# ============================================================
# VERIFIED CORRIDOR PATHS
# ============================================================

CORRIDOR_PATHS = {

    "J1 -> J2": [
        "254053176#2",
        "254053185#1",
        "-254053187",
        "172853384#3",
        "172853384#4",
        "133763768#0",
    ],

    "J2 -> J3": [
        "133763768#1",
        "133763768#2",
        "133763768#5",
        "133763768#6",
    ],

    "J3 -> J4": [
        "150510724#1",
        "150510724#2",
        "150510724#3",
        "150510724#4",
    ],

    "J4 -> J5": [
        "150510724#6",
        "150510724#7",
        "-1311812958",
    ],
}


# ============================================================
# LOAD NETWORK
# ============================================================

print()
print("=" * 110)
print("CORRIDOR MOVEMENT INSPECTION")
print("=" * 110)

print()
print("Network file:")
print(NET_FILE)

print()
print("Loading SUMO network...")

net = sumolib.net.readNet(NET_FILE)

print("Network loaded successfully.")


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_edge_details(edge_id):

    try:
        edge = net.getEdge(edge_id)
    except KeyError:
        print(f"ERROR: Edge not found: {edge_id}")
        return

    print()
    print(f"Edge ID : {edge_id}")
    print(f"Length  : {edge.getLength():.2f} m")
    print(f"Speed   : {edge.getSpeed():.2f} m/s")
    print(f"Lanes   : {len(edge.getLanes())}")

    for lane in edge.getLanes():

        print(
            f"    Lane ID : {lane.getID()} "
            f"| Length = {lane.getLength():.2f} m"
        )


# ============================================================
# INSPECT CORRIDOR PATHS
# ============================================================

print()
print("=" * 110)
print("VERIFIED CORRIDOR PATHS")
print("=" * 110)


for movement, edge_ids in CORRIDOR_PATHS.items():

    print()
    print("-" * 110)
    print(movement)
    print("-" * 110)

    for position, edge_id in enumerate(edge_ids, start=1):

        print()
        print(f"[{position}]")

        print_edge_details(edge_id)


# ============================================================
# IDENTIFY JUNCTION CONNECTIONS
# ============================================================

print()
print("=" * 110)
print("CORRIDOR JUNCTION CONNECTIONS")
print("=" * 110)


for i in range(len(CORRIDOR) - 1):

    source_id = CORRIDOR[i]
    target_id = CORRIDOR[i + 1]

    print()
    print("-" * 110)
    print(f"J{i + 1} -> J{i + 2}")
    print("-" * 110)

    source = net.getNode(source_id)
    target = net.getNode(target_id)

    print(f"Source junction : {source_id}")
    print(f"Target junction : {target_id}")

    print()
    print("Edges leaving source junction:")

    for edge in source.getOutgoing():

        destination = edge.getToNode()

        print(
            f"  {edge.getID()} "
            f"-> {destination.getID()} "
            f"| {edge.getLength():.2f} m"
        )

    print()
    print("Edges entering target junction:")

    for edge in target.getIncoming():

        origin = edge.getFromNode()

        print(
            f"  {edge.getID()} "
            f"<- {origin.getID()} "
            f"| {edge.getLength():.2f} m"
        )


# ============================================================
# FIND TLS CONNECTIONS FOR CORRIDOR EDGES
# ============================================================

print()
print("=" * 110)
print("TLS CONNECTIONS INVOLVING CORRIDOR EDGES")
print("=" * 110)


# Build a set of all corridor edges.

corridor_edge_ids = set()

for path in CORRIDOR_PATHS.values():

    for edge_id in path:

        corridor_edge_ids.add(edge_id)


# SUMO TLS objects

tls_objects = net.getTrafficLights()


for tls in tls_objects:

    tls_id = tls.getID()

    relevant_connections = []

    for connection in tls.getConnections():

        if len(connection) < 2:
            continue

        incoming_lane = connection[0]
        outgoing_lane = connection[1]

        incoming_edge = incoming_lane.getEdge().getID()
        outgoing_edge = outgoing_lane.getEdge().getID()

        if (
            incoming_edge in corridor_edge_ids
            or outgoing_edge in corridor_edge_ids
        ):

            relevant_connections.append(
                (
                    incoming_lane.getID(),
                    outgoing_lane.getID(),
                    incoming_edge,
                    outgoing_edge
                )
            )

    if not relevant_connections:
        continue

    print()
    print("-" * 110)
    print(f"TLS ID : {tls_id}")
    print("-" * 110)

    for (
        incoming_lane_id,
        outgoing_lane_id,
        incoming_edge_id,
        outgoing_edge_id
    ) in relevant_connections:

        print(
            f"  {incoming_lane_id} "
            f"-> {outgoing_lane_id}"
        )

        print(
            f"      Edge: "
            f"{incoming_edge_id} "
            f"-> "
            f"{outgoing_edge_id}"
        )


# ============================================================
# CORRIDOR SUMMARY
# ============================================================

print()
print("=" * 110)
print("CORRIDOR MOVEMENT SUMMARY")
print("=" * 110)

for movement, edge_ids in CORRIDOR_PATHS.items():

    print()
    print(movement)

    print(
        "  Number of edges : "
        f"{len(edge_ids)}"
    )

    total_length = 0.0

    for edge_id in edge_ids:

        try:
            edge = net.getEdge(edge_id)
            total_length += edge.getLength()
        except KeyError:
            pass

    print(
        "  Total distance  : "
        f"{total_length:.2f} m "
        f"({total_length / 1000:.3f} km)"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 110)
print("MOVEMENT INSPECTION COMPLETE")
print("=" * 110)
print()