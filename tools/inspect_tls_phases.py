import os
import sys
import traci


# ============================================================
# PATHS
# ============================================================

SUMO_CONFIG = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "scenario",
        "config",
        "corridor.sumocfg"
    )
)


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
# START SUMO
# ============================================================

print("=" * 100)
print("TLS PHASE PROGRAM INSPECTION")
print("=" * 100)

print("\nSUMO configuration:")
print(SUMO_CONFIG)

print("\nStarting SUMO...")

traci.start([
    "sumo",
    "-c",
    SUMO_CONFIG,
    "--start"
])

print("SUMO started successfully.")


# ============================================================
# INSPECT TLS
# ============================================================

try:

    for tls_id in TLS_IDS:

        print("\n" + "=" * 100)
        print(f"TLS ID : {tls_id}")
        print("=" * 100)

        # ----------------------------------------------------
        # Program information
        # ----------------------------------------------------

        program_ids = traci.trafficlight.getAllProgramLogics(tls_id)

        print(f"\nNumber of programs: {len(program_ids)}")

        for program in program_ids:

            print("\n" + "-" * 90)
            print(f"Program ID : {program.programID}")
            print(f"Type       : {program.type}")
            print(f"Current phase index : {program.currentPhaseIndex}")
            print(f"Number of phases   : {len(program.phases)}")

            # ------------------------------------------------
            # Phases
            # ------------------------------------------------

            for index, phase in enumerate(program.phases):

                print("\n  Phase", index)

                print(f"    Duration : {phase.duration}")
                print(f"    Min dur  : {phase.minDur}")
                print(f"    Max dur  : {phase.maxDur}")
                print(f"    State    : {phase.state}")

        # ----------------------------------------------------
        # Current state
        # ----------------------------------------------------

        print("\nCurrent TLS state:")

        print(
            traci.trafficlight.getRedYellowGreenState(tls_id)
        )

        print(
            "Current phase:",
            traci.trafficlight.getPhase(tls_id)
        )

        print(
            "Current program:",
            traci.trafficlight.getProgram(tls_id)
        )

finally:

    traci.close()


print("\n" + "=" * 100)
print("TLS PHASE INSPECTION COMPLETE")
print("=" * 100)