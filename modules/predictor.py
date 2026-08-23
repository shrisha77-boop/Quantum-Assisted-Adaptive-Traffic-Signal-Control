# Purpose:
# Predict the future traffic state (PCU-weighted density and queue length)
# for each junction using historical traffic data and neighbouring
# junction information.

# ---------------------------------------------------------

# Step 1:
# Receive the current traffic metrics from the
# Traffic Metric Calculation Layer.
# (Current density, queue length and waiting time.)

# ↓

# Step 2:
# Load the historical traffic data of the corresponding
# junction for similar time periods.
# (Used to learn recurring traffic patterns.)

# ↓

# Step 3:
# Receive the latest traffic information from upstream
# junctions that feed the current junction's incoming approaches,
# via the corridor connectivity graph.
# (Outflow rate, signal phase state, queue discharge rate, and
# outgoing density of each upstream junction.)

# ↓

# Step 4:
# Preprocess the collected data.
# • Handle missing values (if any).
# • Arrange the data into time sequences.
# • Normalise/scale the input features if required.

# ↓

# Step 5:
# Prepare the input sequence required by the LSTM model.

# ↓

# Step 6:
# Run the trained LSTM model.

# ↓

# Step 7:
# Predict the future traffic state for the selected
# prediction horizon (e.g., next 20–30 seconds), incorporating
# upstream incoming-flow propagation, travel-time lag, and
# spillback effects.

# Predict:
# • Future PCU-weighted Density
# • Future Queue Length
# • Future Waiting Time
# • Future Inflow Rate

# ↓

# Step 8:
# Validate the predicted values.
# • Ensure values are within acceptable limits.
# • Remove unrealistic predictions if necessary.

# ↓

# Step 9:
# Create a prediction dictionary.

# ↓

# Step 10:
# Send the predicted traffic metrics to the
# Decision Engine.
"""
Predictor
=========

Predicts the near-future traffic state (PCU-weighted density, queue
length, waiting time, and inflow rate) for each approach,
PREDICTION_HORIZON_STEPS decision-cycles ahead, so the Decision Engine
can optimise for where traffic is *heading*, not just where it is right
now.

Per the project specification, prediction is explicitly required to be
spatially aware:

    Prediction(current_junction)
        = f(current_state, upstream_junction_states, corridor_flow_graph)

and must NOT behave as a purely local, history-only predictor:

    Prediction(current_junction) != f(current_junction_history_only)

Concretely, this module incorporates three upstream effects for any
incoming approach whose feeding junction is known (via `corridor_graph`)
and whose current state was supplied (via `upstream_junction_states`):

    A. Incoming flow propagation -- vehicles the upstream junction is
       currently discharging toward us, bounded by its remaining green
       duration.
    B. Travel-time lag -- only discharge that has time to physically
       arrive within the prediction horizon is counted.
    C. Spillback -- an already-congested upstream approach predicts
       elevated future density here, independent of today's discharge
       rate alone.

An approach with no known upstream junction (e.g. a network boundary
entry) simply receives no upstream adjustment and falls back to pure
local extrapolation, which is expected and correct for that case.

Flow implemented (matches the flowchart originally sketched at the top of
this file):

    1. Receive the current traffic metrics from the Traffic Metric
       Calculation Layer.
    2. Load the historical traffic data of this junction for similar
       time periods (used to learn recurring patterns).
    3. Receive the latest traffic information from upstream junctions
       feeding this junction's incoming approaches, via the corridor
       connectivity graph.
    4. Preprocess the collected data (handle missing values, arrange into
       time sequences, normalise).
    5. Prepare the input sequence required by the LSTM model.
    6. Run the trained LSTM model.
    7. Predict the future density/queue/wait/inflow for the prediction
       horizon, then layer in the upstream incoming-flow, travel-time-lag,
       and spillback adjustments (A/B/C above).
    8. Validate the predicted values (clip to plausible ranges).
    9. Build the prediction dictionary.
    10. Return the predicted traffic metrics to the Decision Engine.

LSTM availability
------------------
No trained LSTM model ships with this repo yet (config.py's
LSTM_RETRAIN_EVERY_N_STEPS / LSTM_MIN_LOG_SAMPLES describe an offline batch
retraining schedule that hasn't been wired up to a training script). This
predictor is written so that dropping a model in later is a one-line change:

    Predictor(model_path="models/traffic_lstm.h5")

Until a model is available (or if TensorFlow isn't installed), `predict()`
transparently falls back to a lightweight statistical predictor: a linear
trend fit over the recent history window. This keeps the Decision Engine
functional end-to-end while the LSTM is still being trained, and the
recorded history buffer doubles as the log the eventual training script
would consume. The upstream/corridor-graph adjustment (A/B/C above) is
applied uniformly on top of *either* prediction path, since it is a
required part of the predictor's behaviour regardless of which underlying
model produces the local estimate.

Interface
---------
    predictor.predict(current_metrics, upstream_junction_states=None,
                       corridor_graph=None) -> dict

    current_metrics : dict
        Keyed by direction (subset of config.DIRECTIONS), e.g.:
            {
                "N": {"density": 0.42, "queue_length": 8.0, "wait_time": 12.0},
                ...
            }

    corridor_graph : dict, optional
        Maps this junction's incoming direction to the id of the upstream
        junction feeding that approach (the "Upstream Junction -> Connecting
        Edge -> Current Junction" mapping from the spec, with the connecting
        edge identified implicitly by the direction key), e.g.:
            {"N": "J_upstream_north", "E": "J_upstream_east"}
        A direction with no entry here is treated as having no known
        upstream junction (e.g. a network boundary approach).

    upstream_junction_states : dict, optional
        Keyed by upstream junction id (the values of `corridor_graph`),
        the current state reported by that junction, e.g.:
            {
                "J_upstream_north": {
                    "outflow_rate": 0.4,        # veh/sec leaving toward us
                    "queue_discharge_rate": 0.5,  # veh/sec, defaults to outflow_rate
                    "density": 0.7,             # its outgoing-lane density (spillback signal)
                    "green_remaining": 12.0,    # seconds its green phase has left
                    "travel_time": 8.0,         # seconds for discharge to reach us
                },
                ...
            }
        Fields are individually optional; sensible module-level defaults
        are used for any that are omitted.

    Returns a dict of the same shape as current_metrics, containing the
    predicted "density", "queue_length", "wait_time", and "inflow_rate"
    for PREDICTION_HORIZON_STEPS decision-cycles ahead (matching what
    decision_engine.DecisionEngine and qubo_builder expect as
    `predicted_metrics`).
"""

from collections import deque
from typing import Dict, Optional

import config

# Upper bound used to clip predicted density. Density is PCU / road length
# and is not hard-capped at 1.0 the way DENSITY_THRESHOLD (0.65) is a
# *trigger* rather than a ceiling, so allow generous headroom rather than
# clipping to the threshold itself.
MAX_PLAUSIBLE_DENSITY = 2.0
MAX_PLAUSIBLE_QUEUE_LENGTH = 200.0

# --- Upstream-junction defaults (Section 17: Incoming flow propagation, --
# --- travel-time lag, spillback) -----------------------------------------
# Used only when a given upstream junction's reported state omits these
# fields; keeps the predictor functional even with partial upstream data
# rather than silently ignoring that junction.
DEFAULT_UPSTREAM_TRAVEL_TIME_SECONDS = 15.0
DEFAULT_UPSTREAM_GREEN_REMAINING_SECONDS = 20.0

# How strongly an upstream approach's own congestion (spillback risk)
# inflates our predicted density/queue for the approach it feeds, on top
# of the flow already accounted for via its discharge rate.
SPILLBACK_DENSITY_WEIGHT = 0.3
SPILLBACK_QUEUE_WEIGHT = 0.3


class Predictor:
    """LSTM-backed, upstream-aware traffic predictor with a statistical fallback."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        history_window: Optional[int] = None,
        horizon_steps: Optional[int] = None,
    ):
        self.history_window = history_window or config.HISTORY_WINDOW
        self.horizon_steps = horizon_steps or config.PREDICTION_HORIZON_STEPS

        # predict() is invoked once per Decision Engine cycle (project
        # spec Section 37: ~30 second decision intervals), NOT once per
        # raw SUMO simulation step -- so consecutive history entries are
        # spaced by the decision interval, not by config.SUMO_STEP_LENGTH.
        self.decision_interval_seconds = getattr(
            config,
            "DECISION_INTERVAL_SECONDS",
            getattr(config, "DECISION_INTERVAL", 30.0),
        )

        # config.PREDICTION_HORIZON_STEPS is expressed in raw SUMO
        # simulation steps (config.SUMO_STEP_LENGTH seconds each) -- e.g.
        # 12 steps * 1.0s = a 12-second-ahead prediction target, matching
        # the "next 20-30 seconds" horizon described in the module
        # docstring. This is a real-world time span, independent of how
        # often predict() itself gets called.
        self.horizon_seconds = self.horizon_steps * config.SUMO_STEP_LENGTH

        # The linear-trend extrapolator (_extrapolate) works in "history
        # entries ahead" units, and history entries are spaced by
        # decision_interval_seconds (see above) -- NOT by
        # SUMO_STEP_LENGTH. So the seconds-based horizon must be divided
        # by the decision interval, not by SUMO_STEP_LENGTH, to stay
        # dimensionally consistent. E.g. a 12-second horizon against a
        # 30-second sampling interval means extrapolating only 0.4 of a
        # "step" ahead, not a full step and not 12 full steps.
        self.horizon_entries_ahead = self.horizon_seconds / self.decision_interval_seconds

        # Step 2: per-direction rolling history of observed metrics.
        self.history: Dict[str, deque] = {
            d: deque(maxlen=self.history_window) for d in config.DIRECTIONS
        }

        self.model = None
        self._feature_order = ("density", "queue_length", "wait_time", "inflow_rate")
        if model_path is not None:
            self._load_model(model_path)

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def predict(
        self,
        current_metrics: Dict[str, Dict[str, float]],
        upstream_junction_states: Optional[Dict[str, Dict[str, float]]] = None,
        corridor_graph: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Run one full prediction cycle and return predicted_metrics."""
        # --- Steps 1 & 2: record current metrics into history -----------
        self._record_history(current_metrics)

        # --- Steps 4 & 5: preprocess + build the LSTM input sequence -----
        sequence = self._build_sequence()

        # --- Steps 6 & 7a: predict from local state -----------------------
        if self.model is not None and self._has_enough_history():
            raw_predictions = self._predict_with_lstm(sequence)
        else:
            raw_predictions = self._predict_with_trend()

        # --- Step 7b: layer in upstream flow / lag / spillback effects ----
        # Applied uniformly on top of either prediction path above, since
        # the spec requires upstream-awareness to be a property of the
        # predictor as a whole, not just of the statistical fallback.
        raw_predictions = self._apply_upstream_adjustment(
            raw_predictions, corridor_graph, upstream_junction_states
        )

        # --- Step 8: validate ----------------------------------------------
        predictions = self._validate(raw_predictions)

        # --- Step 9 & 10: return prediction dictionary ----------------------
        return predictions

    # ------------------------------------------------------------------ #
    # Step 2/4: history bookkeeping
    # ------------------------------------------------------------------ #
    def _record_history(self, current_metrics: Dict[str, Dict[str, float]]) -> None:
        for direction in config.DIRECTIONS:
            metrics = current_metrics.get(direction, {})
            last = self.history[direction][-1] if self.history[direction] else None
            density = metrics.get("density")
            queue_length = metrics.get("queue_length")
            wait_time = metrics.get("wait_time")
            inflow_rate = metrics.get("vehicle_count")

            # Every feature falls back to its last known value when the
            # current tick doesn't report it, rather than silently
            # resetting to 0.0 -- consistent handling across all four
            # features (previously inflow_rate alone defaulted straight
            # to 0.0, causing spurious drops whenever "vehicle_count" was
            # momentarily absent from the input).
            if density is None:
                density = last["density"] if last else 0.0
            if queue_length is None:
                queue_length = last["queue_length"] if last else 0.0
            if wait_time is None:
                wait_time = last["wait_time"] if last else 0.0
            if inflow_rate is None:
                inflow_rate = last["inflow_rate"] if last else 0.0

            self.history[direction].append(
                {
                    "density": float(density),
                    "queue_length": float(queue_length),
                    "wait_time": float(wait_time),
                    "inflow_rate": float(inflow_rate),
                }
            )

    def _has_enough_history(self) -> bool:
        return all(len(self.history[d]) >= min(4, self.history_window) for d in config.DIRECTIONS)

    def _build_sequence(self) -> Dict[str, list]:
        """
        Arrange each direction's history into an ordered list of feature
        vectors -- the shape an LSTM expects: (timesteps, features).
        """
        sequence = {}
        for direction in config.DIRECTIONS:
            sequence[direction] = [
                [point[feat] for feat in self._feature_order]
                for point in self.history[direction]
            ]
        return sequence

    # ------------------------------------------------------------------ #
    # Step 6: LSTM model loading + inference (optional)
    # ------------------------------------------------------------------ #
    def _load_model(self, model_path: str) -> None:
        try:
            import os

            if not os.path.exists(model_path):
                self.model = None
                return

            # Lazy import: TensorFlow is a heavy optional dependency; only
            # pay the cost if a model file actually exists to load.
            from tensorflow import keras  # type: ignore

            self.model = keras.models.load_model(model_path)
        except Exception:
            # Missing TensorFlow, corrupt model file, version mismatch,
            # etc. -- fall back to the statistical predictor rather than
            # crashing the whole controller.
            self.model = None

    def _predict_with_lstm(self, sequence: Dict[str, list]) -> Dict[str, Dict[str, float]]:
        import numpy as np  # only imported when the LSTM path is actually used

        predictions: Dict[str, Dict[str, float]] = {}
        for direction in config.DIRECTIONS:
            seq = sequence[direction]
            if len(seq) < 2:
                # Not enough history for this direction specifically --
                # hold the last known value rather than feeding a
                # degenerate sequence to the model.
                last = self.history[direction][-1]
                predictions[direction] = dict(last)
                continue

            x = np.array(seq, dtype="float32")[-self.history_window :]
            x = np.expand_dims(x, axis=0)  # (1, timesteps, features)

            y = self.model.predict(x, verbose=0)[0]

            # The model is currently trained to output density/queue_length
            # only. Fall back to trend extrapolation for wait_time and
            # inflow_rate so this path always returns the same complete
            # feature set as the statistical fallback -- previously these
            # two were simply absent here, which _validate() silently
            # zeroed out downstream instead of raising or estimating them.
            history = list(self.history[direction])
            predictions[direction] = {
                "density": float(y[0]),
                "queue_length": float(y[1]),
                "wait_time": self._extrapolate(history, "wait_time", self.horizon_entries_ahead),
                "inflow_rate": self._extrapolate(history, "inflow_rate", self.horizon_entries_ahead),
            }
        return predictions

    # ------------------------------------------------------------------ #
    # Fallback: linear trend extrapolation
    # ------------------------------------------------------------------ #
    def _predict_with_trend(self) -> Dict[str, Dict[str, float]]:
        predictions: Dict[str, Dict[str, float]] = {}

        for direction in config.DIRECTIONS:
            history = list(self.history[direction])
            predictions[direction] = {
                feat: self._extrapolate(history, feat, self.horizon_entries_ahead)
                for feat in self._feature_order
            }

        return predictions

    @staticmethod
    def _extrapolate(history: list, feature: str, horizon_entries_ahead: float) -> float:
        """
        Fit a simple linear trend (least-squares slope) over the recent
        history for `feature` and extrapolate `horizon_entries_ahead`
        history-entries ahead (history entries are spaced by the decision
        interval -- one per predictor invocation -- not by SUMO_STEP_LENGTH;
        see the `horizon_entries_ahead` computation in __init__). This value
        need not be a whole number: a horizon shorter than one decision
        interval correctly extrapolates only a fraction of a step.
        """
        values = [point[feature] for point in history]
        n = len(values)

        if n == 0:
            return 0.0
        if n == 1:
            return values[0]

        # x = step index (0, 1, 2, ...), one per recorded history entry.
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(values) / n

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
        denominator = sum((x - mean_x) ** 2 for x in xs)
        slope = numerator / denominator if denominator else 0.0

        return values[-1] + slope * horizon_entries_ahead

    # ------------------------------------------------------------------ #
    # Section 17: upstream incoming-flow propagation, travel-time lag,
    # and spillback -- the "NEW CRITICAL INPUT" this predictor must use.
    # ------------------------------------------------------------------ #
    def _apply_upstream_adjustment(
        self,
        raw_predictions: Dict[str, Dict[str, float]],
        corridor_graph: Optional[Dict[str, str]],
        upstream_junction_states: Optional[Dict[str, Dict[str, float]]],
    ) -> Dict[str, Dict[str, float]]:
        if not corridor_graph or not upstream_junction_states:
            # No upstream data supplied for this cycle -- fall back to the
            # local-only estimate already computed. (A direction lacking
            # any entry in corridor_graph, e.g. a network boundary
            # approach, is expected to always take this path.)
            return raw_predictions

        for direction, predicted in raw_predictions.items():
            adjustment = self._upstream_flow_adjustment(
                direction, corridor_graph, upstream_junction_states
            )
            if adjustment is None:
                continue
            predicted["density"] = predicted.get("density", 0.0) + adjustment["density"]
            predicted["queue_length"] = (
                predicted.get("queue_length", 0.0) + adjustment["queue_length"]
            )
            # Prefer the upstream junction's own discharge rate over our
            # noisy local trend for inflow_rate, since it's a more direct
            # causal signal for what's about to arrive -- but only when
            # upstream data was actually available for this direction.
            if adjustment["inflow_rate"] > 0.0:
                predicted["inflow_rate"] = adjustment["inflow_rate"]

        return raw_predictions

    def _upstream_flow_adjustment(
        self,
        direction: str,
        corridor_graph: Dict[str, str],
        upstream_junction_states: Dict[str, Dict[str, float]],
    ) -> Optional[Dict[str, float]]:
        """
        Section 17.A/B/C: incoming flow propagation, travel-time lag, and
        spillback. Returns an additive adjustment to layer on top of the
        local estimate for `direction`, based on the state of whichever
        upstream junction feeds that approach per corridor_graph. Returns
        None if this direction has no known upstream junction, or no
        state was supplied for the one it's mapped to.
        """
        upstream_id = corridor_graph.get(direction)
        if upstream_id is None:
            return None
        upstream = upstream_junction_states.get(upstream_id)
        if not upstream:
            return None

        horizon_seconds = self.horizon_seconds

        # --- B. Travel-time lag: only discharge that has time to reach us
        # within the prediction horizon should be counted. ---
        travel_time = upstream.get("travel_time", DEFAULT_UPSTREAM_TRAVEL_TIME_SECONDS)
        arrivable_window = max(0.0, horizon_seconds - travel_time)

        discharge_rate = upstream.get(
            "queue_discharge_rate", upstream.get("outflow_rate", 0.0)
        )

        if arrivable_window <= 0.0:
            # Nothing discharged upstream now can physically reach us
            # within this horizon.
            incoming_vehicles = 0.0
        else:
            # --- A. Incoming flow propagation: bounded by how much longer
            # the upstream green phase will keep discharging. ---
            green_remaining = upstream.get(
                "green_remaining", DEFAULT_UPSTREAM_GREEN_REMAINING_SECONDS
            )
            discharge_window = min(arrivable_window, max(0.0, green_remaining))
            incoming_vehicles = discharge_rate * discharge_window

        # --- C. Spillback: an already-congested upstream approach means
        # more of its queue will keep backing up onto us, beyond what
        # today's discharge rate alone predicts. ---
        upstream_density = upstream.get("density", 0.0)
        spillback_density = SPILLBACK_DENSITY_WEIGHT * upstream_density
        spillback_queue = SPILLBACK_QUEUE_WEIGHT * incoming_vehicles

        return {
            "density": spillback_density,
            "queue_length": incoming_vehicles + spillback_queue,
            "inflow_rate": discharge_rate,
        }

    # ------------------------------------------------------------------ #
    # Step 8: validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate(predictions: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        validated: Dict[str, Dict[str, float]] = {}
        for direction, metrics in predictions.items():
            density = max(0.0, min(metrics.get("density", 0.0), MAX_PLAUSIBLE_DENSITY))
            queue_length = max(
                0.0, min(metrics.get("queue_length", 0.0), MAX_PLAUSIBLE_QUEUE_LENGTH)
            )
            wait_time = max(0.0, metrics.get("wait_time", 0.0))
            inflow_rate = max(0.0, metrics.get("inflow_rate", 0.0))
            validated[direction] = {
                "density": density,
                "queue_length": queue_length,
                "wait_time": wait_time,
                "inflow_rate": inflow_rate,
            }
        return validated