import os
import sys
import sumolib


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

NET_FILE = os.path.join(
    PROJECT_ROOT,
    "scenario",
    "network",
    "bengaluru.net.xml"
)


# ============================================================
# SELECTED CORRIDOR
# ============================================================

CORRIDOR = [
    "1607769741",

    "cluster_10043935988_10043935989_1070361799",

    "cluster_1070361724_11307884034_11882719580_11882719581_#9more",

    "cluster_13058420509_2571434008_308915752_6137597661",

    "cluster_12074449281_1759977210",
]


# ============================================================
# LOAD NETWORK
# ============================================================

print()
print("=" * 110)
print("CORRIDOR TRAFFIC-LIGHT INSPECTION")
print("=" * 110)

print()
print("Network file:")
print(NET_FILE)

if not os.path.exists(NET_FILE):

    print()
    print("ERROR: Network file not found.")
    print(NET_FILE)
    sys.exit(1)

print()
print("Loading SUMO network...")

net = sumolib.net.readNet(NET_FILE)

print("Network loaded successfully.")


# ============================================================
# BUILD TLS -> CONTROLLED JUNCTION MAPPING
# ============================================================

print()
print("=" * 110)
print("BUILDING TRAFFIC-LIGHT CONTROLLER MAPPING")
print("=" * 110)

tls_mapping = {}


for tls in net.getTrafficLights():

    tls_id = tls.getID()

    controlled_nodes = []

    # --------------------------------------------------------
    # Find nodes controlled by this TLS
    # --------------------------------------------------------

    try:
        nodes = tls.getNodes()

        for node in nodes:
            controlled_nodes.append(node.getID())

    except AttributeError:
        pass

    tls_mapping[tls_id] = controlled_nodes


# ============================================================
# INSPECT SELECTED JUNCTIONS
# ============================================================

results = []


for index, junction_id in enumerate(CORRIDOR, start=1):

    print()
    print()
    print("=" * 110)
    print(f"JUNCTION J{index}")
    print("=" * 110)

    # --------------------------------------------------------
    # Find junction
    # --------------------------------------------------------

    try:

        junction = net.getNode(junction_id)

    except KeyError:

        print()
        print("ERROR: Junction not found:")
        print(junction_id)

        continue

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print()
    print("Junction ID :")
    print(junction.getID())

    x, y = junction.getCoord()

    print()
    print("Position")
    print(f"SUMO X      : {x:.2f}")
    print(f"SUMO Y      : {y:.2f}")

    # --------------------------------------------------------
    # Find TLS controller
    # --------------------------------------------------------

    tls_ids = []

    for tls_id, nodes in tls_mapping.items():

        if junction_id in nodes:

            tls_ids.append(tls_id)

    print()
    print("Traffic Light Controller")

    if tls_ids:

        for tls_id in tls_ids:

            print(f"TLS ID      : {tls_id}")

    else:

        print("TLS ID      : NONE")

    # --------------------------------------------------------
    # Incoming edges
    # --------------------------------------------------------

    incoming = junction.getIncoming()

    print()
    print("-" * 110)
    print(f"INCOMING EDGES ({len(incoming)})")
    print("-" * 110)

    for edge in incoming:

        print()
        print(f"Edge ID     : {edge.getID()}")
        print(f"Length      : {edge.getLength():.2f} m")
        print(f"Speed       : {edge.getSpeed():.2f} m/s")

        lanes = edge.getLanes()

        print(f"Lanes       : {len(lanes)}")

        for lane in lanes:

            print(
                f"    Lane ID : {lane.getID()} "
                f"| Length = {lane.getLength():.2f} m"
            )

    # --------------------------------------------------------
    # Outgoing edges
    # --------------------------------------------------------

    outgoing = junction.getOutgoing()

    print()
    print("-" * 110)
    print(f"OUTGOING EDGES ({len(outgoing)})")
    print("-" * 110)

    for edge in outgoing:

        print()
        print(f"Edge ID     : {edge.getID()}")
        print(f"Length      : {edge.getLength():.2f} m")
        print(f"Speed       : {edge.getSpeed():.2f} m/s")

        lanes = edge.getLanes()

        print(f"Lanes       : {len(lanes)}")

        for lane in lanes:

            print(
                f"    Lane ID : {lane.getID()} "
                f"| Length = {lane.getLength():.2f} m"
            )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append(
        {
            "number": index,
            "junction_id": junction_id,
            "tls_ids": tls_ids,
            "incoming": incoming,
            "outgoing": outgoing,
        }
    )


# ============================================================
# TLS SUMMARY
# ============================================================

print()
print()
print("=" * 110)
print("CORRIDOR TLS SUMMARY")
print("=" * 110)

print()

for result in results:

    number = result["number"]
    junction_id = result["junction_id"]
    tls_ids = result["tls_ids"]

    if tls_ids:

        tls_text = ", ".join(tls_ids)

    else:

        tls_text = "NONE"

    print(
        f"J{number}: {junction_id}"
    )

    print(
        f"     TLS : {tls_text}"
    )

    print()


# ============================================================
# INCOMING EDGE SUMMARY
# ============================================================

print()
print("=" * 110)
print("INCOMING EDGE SUMMARY")
print("=" * 110)

for result in results:

    print()
    print(
        f"J{result['number']}: "
        f"{result['junction_id']}"
    )

    print("-" * 100)

    for edge in result["incoming"]:

        print(
            f"  {edge.getID():<55}"
            f"Length={edge.getLength():>8.2f} m  "
            f"Lanes={len(edge.getLanes())}"
        )


# ============================================================
# OUTGOING EDGE SUMMARY
# ============================================================

print()
print("=" * 110)
print("OUTGOING EDGE SUMMARY")
print("=" * 110)

for result in results:

    print()
    print(
        f"J{result['number']}: "
        f"{result['junction_id']}"
    )

    print("-" * 100)

    for edge in result["outgoing"]:

        print(
            f"  {edge.getID():<55}"
            f"Length={edge.getLength():>8.2f} m  "
            f"Lanes={len(edge.getLanes())}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 110)
print("FINAL CORRIDOR SUMMARY")
print("=" * 110)

print()

print("Selected corridor:")

for index, junction_id in enumerate(CORRIDOR, start=1):

    print(
        f"J{index} = {junction_id}"
    )

print()

print(
    f"Successfully inspected: "
    f"{len(results)}/5 junctions"
)

print()
print("=" * 110)
print("INSPECTION COMPLETE")
print("=" * 110)
print()