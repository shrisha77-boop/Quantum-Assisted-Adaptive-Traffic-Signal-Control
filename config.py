"""
Central configuration for the quantum-assisted adaptive traffic signal system.
Tune these constants instead of hardcoding values inside modules.
"""

# --- Original hard-rule thresholds (kept from baseline algorithm) ---
MAX_WAIT_SECONDS = 180       # urgency = 1.0 at this wait time (must be > STARVATION_THRESHOLD)
STARVATION_THRESHOLD = 120   # force-override any phase waiting > this many seconds
DENSITY_THRESHOLD = 0.65     # high-density override triggers above this
MIN_GREEN_TIME = 15
MAX_GREEN_TIME = 60

# --- Signal transition safety durations ---
YELLOW_TIME_SECONDS = 3.0
ALL_RED_TIME_SECONDS = 2.0

# --- Single-Solver Traffic Load Thresholds (Section 29/30) ---
LOW_TRAFFIC_THRESHOLD = 20
HIGH_TRAFFIC_THRESHOLD = 20


DIRECTIONS = ["N", "E", "S", "W"]

# Phase capacity model: each phase runs its two parallel approaches
# simultaneously (NS = North+South green together, EW = East+West green
# together), matching the SUMO baseline's physical capacity. The controller
# must choose between "NS" and "EW" as a whole phase, never a single
# direction in isolation, or effective capacity is cut in half.
PHASES = ["NS", "EW"]
PHASE_MAP = {
    "NS": ["N", "S"],
    "EW": ["E", "W"]
}

# --- Per-approach incident isolation ---
# When True, an incident-isolated approach is held red on its own while its
# parallel partner in the same phase (see PHASE_MAP) keeps flowing on green.
# This avoids losing an entire phase's green time to a single blocked
# approach.
ENABLE_PER_APPROACH_ISOLATION = True
ISOLATED_APPROACH_STATE = "red"   # state forced on an isolated approach
ISOLATION_RELEASE_REQUIRES_CLEARANCE = True  # only rejoin phase after incident-clearance checks pass (see INCIDENT_* below)

# --- Prediction ---
PREDICTION_HORIZON_STEPS = 12
HISTORY_WINDOW = 60

# --- PCU (Passenger Car Unit) weighting for mixed-vehicle density ---
# Converts raw vehicle counts into a comparable "car-equivalent" load.
PCU_WEIGHTS = {
    "motorcycle": 0.5,
    "passenger": 1.0,     # car (SUMO vClass naming)
    "bus": 3.0,
    "truck": 3.0,
    "bicycle": 0.3,
    # Aliases matching this project's actual SUMO <vType id="..."> values
    # (scenario/routes/corridor.rou.xml), which traci.vehicle.getTypeID()
    # returns verbatim -- these are NOT SUMO vClass names, so they need
    # their own entries with the same PCU factors as their vClass
    # equivalents above (car == passenger, bike == motorcycle).
    "car": 1.0,
    "bike": 0.5,
    "default": 1.0,
}
GPS_FUSION_WEIGHT = 0.4   # blend factor: final_density = (1-w)*sensor + w*gps_estimate

# --- Emergency detection ---
YOLO_CONF_THRESHOLD = 0.6   # min confidence to accept a YOLO emergency-vehicle detection
EMERGENCY_HOLD_TIMEOUT = 14

# --- Incident isolation / clearance ---
INCIDENT_CV_ZERO_OUTFLOW_TICKS = 6
INCIDENT_CLEAR_CAPACITY_PCT = 0.70   # outflow must exceed 70% of road capacity...
INCIDENT_CLEAR_CONSECUTIVE_CYCLES = 3  # ...for 3 consecutive cycles to confirm clearance

# --- QUBO / three-tier solver ---
QUBO_ONE_HOT_PENALTY = 8.0
QUBO_COORDINATION_BONUS = 2.0
QUANTUM_QPU_TIMEOUT = 3.0        # D-Wave QPU budget
SIMULATED_ANNEALING_TIMEOUT = 1.5  # neal fallback budget
NUM_ANNEALING_READS = 200
USE_REAL_QPU = False             # set True + configure D-Wave Leap token to use real hardware

# --- RL tuner: OFFLINE batch updates, not per-cycle ---
RL_LEARNING_RATE = 0.1
RL_DISCOUNT = 0.9
RL_EXPLORATION_EPS = 0.1
RL_RETRAIN_EVERY_N_CYCLES = 50    # batch update interval

# --- LSTM: retrained DAILY on accumulated logs, not online per-step ---
LSTM_RETRAIN_EVERY_N_STEPS = 86400  # e.g. 1 sim-day at 1s/step; adjust to your step length
LSTM_MIN_LOG_SAMPLES = 200

# --- SUMO ---
SUMO_STEP_LENGTH = 1.0