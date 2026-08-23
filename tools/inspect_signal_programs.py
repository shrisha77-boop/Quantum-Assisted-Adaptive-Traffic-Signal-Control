import os
import sys
import sumolib


# ============================================================
# PATH
# ============================================================

NETWORK_FILE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "scenario",
        "network",
        "bengaluru.net.xml"
    )
)


# ============================================================
# LOAD NETWORK
# ============================================================

print("=" * 100)
print("TRAFFIC SIGNAL PROGRAM INSPECTION")
print("=" * 100)

print(f"\nNetwork file:")
print(NETWORK_FILE)

print("\nLoading SUMO network...")

net = sumolib.net.readNet(NETWORK_FILE)

print("Network loaded successfully.")


# ============================================================
# TLS IDS
# ============================================================

TLS_IDS = [
    "1607769741",
    "cluster_10043935988_10043935989_1070361799",
    "cluster_1070361724_11307884034_11882719580_11882719581_#9more",
    "cluster_13058420509_2571434008_308915752_6137597661",
    "cluster_12074449281_1759977210",
]


# ============================================================
# INSPECT
# ============================================================

print("\n" + "=" * 100)
print("CORRIDOR TRAFFIC LIGHTS")
print("=" * 100)

for tls_id in TLS_IDS:

    print("\n" + "-" * 100)
    print(f"TLS ID : {tls_id}")
    print("-" * 100)

    tls = net.getTLS(tls_id)

    if tls is None:
        print("ERROR: TLS not found.")
        continue

    print("TLS found successfully.")

    connections = tls.getConnections()

    print(f"\nNumber of TLS connections: {len(connections)}")

    for connection in connections:

        from_lane = connection[0]
        to_lane = connection[1]

        print(
            f"  {from_lane.getID()} -> {to_lane.getID()}"
        )


print("\n" + "=" * 100)
print("INSPECTION COMPLETE")
print("=" * 100)