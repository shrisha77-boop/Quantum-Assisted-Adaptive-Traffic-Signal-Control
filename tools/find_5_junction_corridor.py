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

MAX_DISTANCE_KM = 3.0

MIN_INCOMING = 3
MIN_OUTGOING = 3


def distance_km(lat1, lon1, lat2, lon2):
    """
    Approximate geographic distance using the Haversine formula.
    """

    R = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


print("\nLoading SUMO network...")

net = sumolib.net.readNet(NET_FILE)

print("Network loaded successfully.\n")


# ------------------------------------------------------------
# Find suitable traffic-light junctions
# ------------------------------------------------------------

junctions = []

for node in net.getNodes():

    if node.getType() != "traffic_light":
        continue

    incoming = node.getIncoming()
    outgoing = node.getOutgoing()

    if len(incoming) < MIN_INCOMING:
        continue

    if len(outgoing) < MIN_OUTGOING:
        continue

    x, y = node.getCoord()

    lon, lat = net.convertXY2LonLat(x, y)

    distance = distance_km(
        TARGET_LAT,
        TARGET_LON,
        lat,
        lon
    )

    if distance > MAX_DISTANCE_KM:
        continue

    junctions.append({
        "id": node.getID(),
        "lat": lat,
        "lon": lon,
        "incoming": len(incoming),
        "outgoing": len(outgoing),
        "distance": distance,
        "node": node
    })


# ------------------------------------------------------------
# Sort by distance from target
# ------------------------------------------------------------

junctions.sort(key=lambda j: j["distance"])


print("=" * 120)
print("SUITABLE SIGNALIZED JUNCTIONS")
print("=" * 120)

for i, j in enumerate(junctions, 1):

    print(
        f"{i:2d}. "
        f"{j['id']:<70} "
        f"{j['lat']:.6f} "
        f"{j['lon']:.6f} "
        f"In={j['incoming']} "
        f"Out={j['outgoing']}"
    )


# ------------------------------------------------------------
# Find physically close junction pairs
# ------------------------------------------------------------

print("\n")
print("=" * 120)
print("CLOSE JUNCTION PAIRS")
print("=" * 120)

pairs = []

for i in range(len(junctions)):

    for j in range(i + 1, len(junctions)):

        a = junctions[i]
        b = junctions[j]

        d = distance_km(
            a["lat"],
            a["lon"],
            b["lat"],
            b["lon"]
        )

        # Candidate corridor spacing:
        # roughly 300 m to 1500 m
        if 0.3 <= d <= 1.5:

            pairs.append({
                "a": a,
                "b": b,
                "distance": d
            })


pairs.sort(key=lambda p: p["distance"])


for p in pairs:

    print(
        f"{p['distance']:.3f} km : "
        f"{p['a']['id']} "
        f"<-> "
        f"{p['b']['id']}"
    )


# ------------------------------------------------------------
# Look for groups of 5 nearby junctions
# ------------------------------------------------------------

print("\n")
print("=" * 120)
print("5-JUNCTION CORRIDOR GROUPS")
print("=" * 120)


groups = []

for start in range(len(junctions)):

    start_junction = junctions[start]

    group = [start_junction]

    for candidate in junctions:

        if candidate is start_junction:
            continue

        if len(group) >= 5:
            break

        d = distance_km(
            start_junction["lat"],
            start_junction["lon"],
            candidate["lat"],
            candidate["lon"]
        )

        if d <= 2.0:
            group.append(candidate)

    if len(group) == 5:

        # Sort geographically by longitude/latitude
        group_sorted = sorted(
            group,
            key=lambda j: (j["lon"], j["lat"])
        )

        groups.append(group_sorted)


# Remove duplicate groups

unique_groups = []

seen = set()

for group in groups:

    ids = tuple(sorted(j["id"] for j in group))

    if ids in seen:
        continue

    seen.add(ids)
    unique_groups.append(group)


# ------------------------------------------------------------
# Print groups
# ------------------------------------------------------------

for number, group in enumerate(unique_groups[:10], 1):

    print("\n")
    print(f"CORRIDOR {number}")
    print("-" * 100)

    for index, j in enumerate(group, 1):

        print(
            f"J{index}: "
            f"{j['id']}\n"
            f"    Latitude : {j['lat']:.6f}\n"
            f"    Longitude: {j['lon']:.6f}\n"
            f"    Incoming : {j['incoming']}\n"
            f"    Outgoing : {j['outgoing']}"
        )


print("\n")
print("=" * 120)
print("Analysis complete.")
print("=" * 120)