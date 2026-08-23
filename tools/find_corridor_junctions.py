import os
import sys
import math

SUMO_HOME = os.environ.get("SUMO_HOME")

if not SUMO_HOME:
    raise RuntimeError("SUMO_HOME environment variable is not set.")

sys.path.append(os.path.join(SUMO_HOME, "tools"))

import sumolib


NET_FILE = "scenario/network/bengaluru.net.xml"

TARGET_LAT = 12.925752
TARGET_LON = 77.623150

MAX_DISTANCE = 0.015


print("\nLoading SUMO network...")
net = sumolib.net.readNet(NET_FILE)
print("Network loaded successfully.\n")


candidates = []


for node in net.getNodes():

    if node.getType() != "traffic_light":
        continue

    incoming = node.getIncoming()
    outgoing = node.getOutgoing()

    # Ignore tiny traffic-light components
    if len(incoming) < 3 or len(outgoing) < 3:
        continue

    x, y = node.getCoord()

    lon, lat = net.convertXY2LonLat(x, y)

    distance = math.sqrt(
        (lat - TARGET_LAT) ** 2 +
        (lon - TARGET_LON) ** 2
    )

    if distance > MAX_DISTANCE:
        continue

    # --------------------------------------------------------
    # Collect road names
    # --------------------------------------------------------

    road_names = set()

    for edge in incoming:

        name = edge.getName()

        if name:
            road_names.add(name)

    for edge in outgoing:

        name = edge.getName()

        if name:
            road_names.add(name)

    candidates.append({
        "id": node.getID(),
        "lat": lat,
        "lon": lon,
        "incoming": len(incoming),
        "outgoing": len(outgoing),
        "distance": distance,
        "roads": sorted(road_names)
    })


candidates.sort(key=lambda x: x["distance"])


print("=" * 120)
print("SIGNALIZED JUNCTION CANDIDATES")
print("=" * 120)

for i, j in enumerate(candidates, start=1):

    print("\n" + "-" * 120)

    print(f"Candidate : {i}")
    print(f"ID        : {j['id']}")
    print(f"Latitude  : {j['lat']:.6f}")
    print(f"Longitude : {j['lon']:.6f}")
    print(f"Incoming  : {j['incoming']}")
    print(f"Outgoing  : {j['outgoing']}")

    print("Roads:")

    if j["roads"]:
        for road in j["roads"]:
            print(f"   - {road}")
    else:
        print("   - [road name unavailable]")


print("\n")
print("=" * 120)
print("TOTAL CANDIDATES:", len(candidates))
print("=" * 120)