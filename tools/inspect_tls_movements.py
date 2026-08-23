import os
import traci


# ============================================================
# PATH
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
# CORRIDOR EDGES
# ============================================================

FORWARD_EDGES = {
    "254053176#2",
    "254053185#1",
    "-254053187",
    "172853384#3",
    "172853384#4",
    "133763768#0",
    "133763768#1",
    "133763768#2",
    "133763768#5",
    "133763768#6",
    "150510724#1",
    "150510724#2",
    "150510724#3",
    "150510724#4",
    "150510724#6",
    "150510724#7",
    "-1311812958",
}


REVERSE_EDGES = {
    "-631197284#3",
    "-631197284#2",
    "-631197284#1",
    "27445936#7",
    "150510724#4",
    "24231604#0",
    "24231604#1",
    "24231604#2",
    "24231604#3",
    "24231604#4",
    "1387944292",
    "111814614#0",
    "111814614#1",
    "111814614#2",
    "111814614#5",
    "111814614#6",
    "111814614#8",
    "172853384#4",
    "464465166#0",
    "-1322924307",
    "-559785241#1",
    "-559785241#0",
    "-254053185#1",
    "-254053176#2",
}


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


TLS_NAMES = {
    TLS_IDS[0]: "J1",
    TLS_IDS[1]: "J2",
    TLS_IDS[2]: "J3",
    TLS_IDS[3]: "J4",
    TLS_IDS[4]: "J5",
}


# ============================================================
# START SUMO
# ============================================================

print("=" * 110)
print("TLS MOVEMENT AND PHASE MAPPING")
print("=" * 110)

print("\nStarting SUMO...")

traci.start([
    "sumo",
    "-c",
    SUMO_CONFIG,
    "--start"
])

print("SUMO started successfully.")


# ============================================================
# INSPECTION
# ============================================================

try:

    for tls_id in TLS_IDS:

        junction = TLS_NAMES[tls_id]

        print("\n")
        print("=" * 110)
        print(f"{junction}  |  TLS ID: {tls_id}")
        print("=" * 110)

        # ----------------------------------------------------
        # Controlled links
        # ----------------------------------------------------

        controlled_links = traci.trafficlight.getControlledLinks(
            tls_id
        )

        # ----------------------------------------------------
        # Current program
        # ----------------------------------------------------

        program_id = traci.trafficlight.getProgram(tls_id)

        programs = traci.trafficlight.getAllProgramLogics(
            tls_id
        )

        selected_program = None

        for program in programs:

            if program.programID == program_id:
                selected_program = program
                break

        if selected_program is None:

            print("Could not find active program.")
            continue

        print(f"\nActive program: {program_id}")
        print(
            f"Number of phases: "
            f"{len(selected_program.phases)}"
        )

        # ----------------------------------------------------
        # Print mapping
        # ----------------------------------------------------

        print("\n" + "-" * 110)
        print("LINK MAPPING")
        print("-" * 110)

        for index, links in enumerate(controlled_links):

            print(f"\nLINK INDEX {index}")

            for link in links:

                incoming_lane = link[0]
                outgoing_lane = link[1]

                incoming_edge = (
                    incoming_lane.split("_")[0]
                )

                outgoing_edge = (
                    outgoing_lane.split("_")[0]
                )

                if incoming_edge in FORWARD_EDGES:

                    direction = "FORWARD"

                elif incoming_edge in REVERSE_EDGES:

                    direction = "REVERSE"

                else:

                    direction = "OTHER"

                print(
                    f"  {incoming_lane}"
                    f" -> "
                    f"{outgoing_lane}"
                )

                print(
                    f"  Incoming edge : "
                    f"{incoming_edge}"
                )

                print(
                    f"  Outgoing edge : "
                    f"{outgoing_edge}"
                )

                print(
                    f"  Direction     : "
                    f"{direction}"
                )

        # ----------------------------------------------------
        # Phase states
        # ----------------------------------------------------

        print("\n" + "-" * 110)
        print("PHASE STATES")
        print("-" * 110)

        for phase_index, phase in enumerate(
            selected_program.phases
        ):

            print(
                f"\nPhase {phase_index}"
            )

            print(
                f"Duration : {phase.duration}s"
            )

            print(
                f"State    : {phase.state}"
            )

            print(
                f"Length   : {len(phase.state)}"
            )


finally:

    traci.close()


print("\n")
print("=" * 110)
print("TLS MOVEMENT MAPPING COMPLETE")
print("=" * 110)