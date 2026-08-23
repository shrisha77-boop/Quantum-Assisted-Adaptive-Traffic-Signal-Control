import os
import sys
import csv
import traci


# ============================================================
# CONFIGURATION
# ============================================================

SIMULATION_END = 3600       # Run for 1 hour
COLLECTION_INTERVAL = 5     # Collect vehicle data every 5 seconds


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUMO_CONFIG = os.path.join(
    BASE_DIR,
    "scenario",
    "config",
    "corridor.sumocfg"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "simulation",
    "data",
    "vehicle_data.csv"
)


# ============================================================
# CHECK SUMO_HOME
# ============================================================

if "SUMO_HOME" in os.environ:

    tools = os.path.join(
        os.environ["SUMO_HOME"],
        "tools"
    )

    if tools not in sys.path:
        sys.path.append(tools)

else:

    print("ERROR: SUMO_HOME environment variable is not set.")
    print("Please set SUMO_HOME to your SUMO installation directory.")
    sys.exit(1)


# ============================================================
# CHECK CONFIGURATION FILE
# ============================================================

if not os.path.exists(SUMO_CONFIG):

    print("ERROR: SUMO configuration file not found:")
    print(SUMO_CONFIG)

    sys.exit(1)


# ============================================================
# START SUMO
# ============================================================

sumo_binary = "sumo"

sumo_cmd = [
    sumo_binary,
    "-c",
    SUMO_CONFIG,
    "--start"
]


print("=" * 70)
print("TRAFFIC DATA COLLECTION")
print("=" * 70)

print()
print("SUMO configuration:")
print(SUMO_CONFIG)

print()
print("Output file:")
print(OUTPUT_FILE)

print()
print("Simulation duration:")
print(f"{SIMULATION_END} seconds")

print()
print("Collection interval:")
print(f"{COLLECTION_INTERVAL} seconds")

print()
print("Starting SUMO...")

traci.start(sumo_cmd)

print("SUMO started successfully.")
print()


# ============================================================
# PREPARE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# ============================================================
# PREPARE CSV FILE
# ============================================================

csv_file = open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(csv_file)

writer.writerow([
    "time",
    "vehicle_id",
    "vehicle_type",
    "edge_id",
    "lane_id",
    "speed",
    "waiting_time",
    "position_x",
    "position_y"
])


# ============================================================
# SIMULATION VARIABLES
# ============================================================

step = 0
vehicle_records = 0
last_progress_time = -1


# ============================================================
# SIMULATION LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # Advance SUMO by one simulation step
        # ----------------------------------------------------

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        step += 1


        # ----------------------------------------------------
        # Stop exactly at the required simulation duration
        # ----------------------------------------------------

        if current_time > SIMULATION_END:
            break


        # ----------------------------------------------------
        # Collect data only every COLLECTION_INTERVAL seconds
        # ----------------------------------------------------

        if int(current_time) % COLLECTION_INTERVAL != 0:
            continue


        # ----------------------------------------------------
        # Get active vehicles
        # ----------------------------------------------------

        vehicle_ids = traci.vehicle.getIDList()


        # ----------------------------------------------------
        # Collect vehicle-level data
        # ----------------------------------------------------

        for vehicle_id in vehicle_ids:

            try:

                vehicle_type = traci.vehicle.getTypeID(
                    vehicle_id
                )

                edge_id = traci.vehicle.getRoadID(
                    vehicle_id
                )

                lane_id = traci.vehicle.getLaneID(
                    vehicle_id
                )

                speed = traci.vehicle.getSpeed(
                    vehicle_id
                )

                waiting_time = traci.vehicle.getWaitingTime(
                    vehicle_id
                )

                position = traci.vehicle.getPosition(
                    vehicle_id
                )


                # ------------------------------------------------
                # Write record
                # ------------------------------------------------

                writer.writerow([
                    current_time,
                    vehicle_id,
                    vehicle_type,
                    edge_id,
                    lane_id,
                    round(speed, 3),
                    round(waiting_time, 3),
                    round(position[0], 3),
                    round(position[1], 3)
                ])

                vehicle_records += 1


            except traci.TraCIException:

                # Vehicle may disappear between getting the
                # vehicle list and querying its information.
                continue


        # ----------------------------------------------------
        # Flush data to disk
        # ----------------------------------------------------

        csv_file.flush()


        # ----------------------------------------------------
        # Progress display every 100 seconds
        # ----------------------------------------------------

        if (
            int(current_time) % 100 == 0
            and int(current_time) != last_progress_time
        ):

            print(
                f"Time: {current_time:6.0f}s | "
                f"Vehicles: {len(vehicle_ids):4d} | "
                f"Records: {vehicle_records}"
            )

            last_progress_time = int(current_time)


# ============================================================
# CLEANUP
# ============================================================

finally:

    csv_file.close()

    try:
        traci.close()
    except Exception:
        pass


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("DATA COLLECTION COMPLETE")
print("=" * 70)

print()
print(f"Simulation steps : {step}")
print(f"Vehicle records  : {vehicle_records}")
print(f"Output file      : {OUTPUT_FILE}")

print()
print("=" * 70)