# Work to be done

# Receive:
# Selected phase.
# Green time.
# Check whether the selected phase differs from the current phase.
# Apply yellow transition if required.
# Activate the new green phase.
# Maintain the green phase for the calculated duration.
# Return control after the decision interval.
"""
Signal Controller
==================

The last stop in the pipeline: takes the Decision Engine's chosen phase and
green duration and actually drives the SUMO traffic light (via TraCI),
handling the yellow / all-red safety transition whenever the phase
actually changes.

Flow implemented (matches the flowchart originally sketched at the top of
this file):

    1. Receive the selected phase and green time.
    2. Check whether the selected phase differs from the current phase.
    3. Apply a yellow transition if required (never skip straight from
       green to green on a different phase -- that's a hard safety rule).
    4. Activate the new green phase.
    5. Maintain the green phase for the calculated duration.
    6. Return control after the decision interval.

Why this needs its own state machine
-------------------------------------
Per main_controller.py's pseudocode, the Signal Controller is called every
single simulation step, and decision_engine.DecisionEngine re-runs its
optimisation every step too. That means `apply()` can be called dozens of
times while a single green phase is still supposed to be running. This
module is what actually enforces "maintain the green phase for the
calculated duration": it tracks its own internal state (GREEN / YELLOW /
ALL_RED) and elapsed time, and only actually starts a phase change once the
currently-committed green interval has been served (or `force=True` is
passed for an emergency override) -- even if it's asked to switch every
single step. A yellow -> all-red transition is always inserted before a new
phase turns green; this is never skipped, including for emergencies.

Interface
---------
    signal_controller.apply(phase, green_time, isolated_approaches=None, force=False) -> dict

This is exactly what decision_engine.DecisionEngine._dispatch() calls
(with just `phase` and `green_time`), so it works out of the box. The two
extra keyword arguments are optional, forward-compatible hooks:

    isolated_approaches : set[str], optional
        Directions currently held red per road_isolation_manager.py /
        config.ISOLATED_APPROACH_STATE, even if their phase partner is
        flowing (see config.ENABLE_PER_APPROACH_ISOLATION). If omitted,
        the last value passed in (or set via `set_isolated_approaches`)
        is reused.
    force : bool, optional
        Skip the "must serve the committed green time first" wait (e.g.
        for decision.emergency / decision.starvation_override). The
        yellow/all-red safety transition is still never skipped.

You can wire emergencies straight through by calling:

    signal_controller.apply(decision.phase, decision.green_time,
                             isolated_approaches=decision.isolated_approaches,
                             force=decision.emergency or decision.starvation_override)

from main_controller.py instead of decision_engine's default two-argument
dispatch, if you want emergency vehicles to preempt a still-committed green
immediately rather than waiting for it to finish.

Traffic-light wiring
---------------------
Building the exact "GrYr..." state string SUMO expects requires knowing
which controlled-link indices belong to which approach direction -- this is
network-specific (from the .net.xml), so it's supplied once at
construction time rather than guessed:

    SignalController(
        tls_id="junction1",
        direction_signal_groups={
            "N": [0, 1], "E": [2, 3], "S": [4, 5], "W": [6, 7]
        },
    )

`direction_signal_groups` values are the traffic-light link indices that
should be green when that direction has right of way (typically obtained
once via `traci.trafficlight.getControlledLinks(tls_id)`).

`traci_module` can be injected for testing without a running SUMO instance
(pass any object exposing `trafficlight.setRedYellowGreenState(tls_id, state)`);
production code leaves it as None and the real `traci` package is imported
lazily on first use.
"""

from typing import Dict, List, Optional, Set

import config

# --- Safety-transition timings -------------------------------------------
YELLOW_TIME_SECONDS = getattr(config, "YELLOW_TIME_SECONDS", 3.0)
ALL_RED_TIME_SECONDS = getattr(config, "ALL_RED_TIME_SECONDS", 2.0)

STATE_GREEN = "GREEN"
STATE_YELLOW = "YELLOW"
STATE_ALL_RED = "ALL_RED"


class SignalController:
    """Drives one intersection's traffic light through TraCI."""

    def __init__(
        self,
        tls_id: str,
        direction_signal_groups: Dict[str, List[int]],
        num_links: Optional[int] = None,
        traci_module=None,
        phase_state_map: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self.tls_id = tls_id
        self.direction_signal_groups = direction_signal_groups
        self.phase_state_map = phase_state_map or {}
        self.num_links = num_links or (
            max((idx for indices in direction_signal_groups.values() for idx in indices), default=-1) + 1
        )
        self._traci = traci_module  # lazily imported real `traci` if None

        # --- internal state machine ---
        self.current_phase = config.PHASES[0]
        self.state = STATE_GREEN
        self.time_in_state = 0.0
        self.committed_green_time = config.MIN_GREEN_TIME
        self.pending_phase: Optional[str] = None
        self.pending_green_time: Optional[float] = None
        self.isolated_approaches: Set[str] = set()

        # Reverse lookup (link index -> direction), built once here instead
        # of scanning direction_signal_groups from scratch for every link on
        # every simulation step inside _build_state_string()'s fallback path.
        self._index_to_direction: Dict[int, str] = {
            index: direction
            for direction, indices in direction_signal_groups.items()
            for index in indices
        }

        self._apply_state_to_traci()


    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def apply(
        self,
        phase: str,
        green_time: float,
        isolated_approaches: Optional[Set[str]] = None,
        force: bool = False,
    ) -> Dict:
        """
        Advance the signal state machine by one simulation step, honouring
        (or queuing) the requested phase/green_time.

        Returns a small status dict, mainly useful for logging/debugging:
            {"state": "GREEN"|"YELLOW"|"ALL_RED",
             "active_phase": "NS"|"EW",
             "phase_changed": bool}
        """
        if phase not in config.PHASES:
            raise ValueError(f"Unknown phase {phase!r}; expected one of {config.PHASES}")

        green_time = self._clamp_green_time(green_time)
        if isolated_approaches is not None:
            self.isolated_approaches = set(isolated_approaches)

        dt = config.SUMO_STEP_LENGTH
        phase_changed = False

        if self.state == STATE_GREEN:
            phase_changed = self._handle_green(phase, green_time, dt, force)
        elif self.state == STATE_YELLOW:
            self._handle_yellow(phase, green_time, dt)
        elif self.state == STATE_ALL_RED:
            phase_changed = self._handle_all_red(phase, green_time, dt)

        self._apply_state_to_traci()

        return {
            "state": self.state,
            "active_phase": self.current_phase,
            "phase_changed": phase_changed,
        }

    def set_isolated_approaches(self, isolated_approaches: Set[str]) -> None:
        """
        Update which approaches are currently held red, independent of the
        next `apply()` call (e.g. call this from road_isolation_manager.py
        directly if you'd rather not thread it through every apply() call).
        """
        self.isolated_approaches = set(isolated_approaches)

    def get_status(self) -> Dict:
        """Snapshot of the controller's current state, for logging/tests."""
        return {
            "tls_id": self.tls_id,
            "state": self.state,
            "active_phase": self.current_phase,
            "time_in_state": self.time_in_state,
            "committed_green_time": self.committed_green_time,
            "isolated_approaches": set(self.isolated_approaches),
        }

    # ------------------------------------------------------------------ #
    # State machine: GREEN
    # ------------------------------------------------------------------ #
    def _handle_green(self, phase: str, green_time: float, dt: float, force: bool) -> bool:
        if phase == self.current_phase:
            self.time_in_state += dt
            if self.time_in_state >= self.committed_green_time:
                # This phase has served its committed duration but the
                # Decision Engine still wants it -- start a fresh green
                # interval rather than forcing a pointless transition.
                self.time_in_state = 0.0
                self.committed_green_time = green_time
            else:
                # Allow the Decision Engine to extend the commitment, but
                # never silently shrink a green interval that's already
                # partway through.
                self.committed_green_time = max(self.committed_green_time, green_time)
            return False

        # A different phase is being requested.
        commitment_served = self.time_in_state >= self.committed_green_time
        if not commitment_served and not force:
            # Still owe the current phase its green time -- keep going.
            self.time_in_state += dt
            return False

        # Commitment served (or forced override) -- start the safety
        # transition. The yellow/all-red interval is never skipped, even
        # when force=True.
        self._begin_yellow(phase, green_time)
        return True

    # ------------------------------------------------------------------ #
    # State machine: YELLOW
    # ------------------------------------------------------------------ #
    def _begin_yellow(self, next_phase: str, next_green_time: float) -> None:
        self.state = STATE_YELLOW
        self.time_in_state = 0.0
        self.pending_phase = next_phase
        self.pending_green_time = next_green_time

    def _handle_yellow(self, phase: str, green_time: float, dt: float) -> None:
        # Keep the pending target in sync with the latest decision. This
        # never shortens or skips the yellow interval -- only *what* we
        # transition to once it finishes can change. Without this, a phase
        # request that arrives mid-yellow (e.g. an emergency vehicle whose
        # required phase changes after the transition already started)
        # would be silently dropped for a full yellow+all-red cycle: the
        # controller would still activate whatever phase happened to be
        # requested at the instant the transition began.
        self.pending_phase = phase
        self.pending_green_time = green_time
        self.time_in_state += dt
        if self.time_in_state >= YELLOW_TIME_SECONDS:
            self._begin_all_red()

    # ------------------------------------------------------------------ #
    # State machine: ALL_RED
    # ------------------------------------------------------------------ #
    def _begin_all_red(self) -> None:
        self.state = STATE_ALL_RED
        self.time_in_state = 0.0

    def _handle_all_red(self, phase: str, green_time: float, dt: float) -> bool:
        self.pending_phase = phase
        self.pending_green_time = green_time
        self.time_in_state += dt
        if self.time_in_state >= ALL_RED_TIME_SECONDS:
            self._activate_green(self.pending_phase, self.pending_green_time)
            return True
        return False

    def _activate_green(self, phase: str, green_time: float) -> None:
        self.current_phase = phase
        self.committed_green_time = green_time
        self.state = STATE_GREEN
        self.time_in_state = 0.0
        self.pending_phase = None
        self.pending_green_time = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clamp_green_time(green_time: float) -> float:
        return max(config.MIN_GREEN_TIME, min(config.MAX_GREEN_TIME, green_time))

    def _green_directions(self, phase: str) -> Set[str]:
        """Directions that should show green for `phase`, isolation-aware."""
        return set(config.PHASE_MAP[phase]) - self.isolated_approaches

    def _build_state_string(self) -> str:
        """Build the SUMO RYG state string for the controller's current state."""
        if self.phase_state_map:
            phase_name = self.current_phase
            phase_info = self.phase_state_map.get(phase_name, {})
            if self.state == STATE_GREEN:
                base_state = phase_info.get("green", "")
            elif self.state == STATE_YELLOW:
                base_state = phase_info.get("yellow", "")
            else:  # ALL_RED
                base_state = phase_info.get("all_red", "") or ("r" * len(phase_info.get("green", "")))

            if base_state:
                if not self.isolated_approaches:
                    return base_state
                chars = list(base_state)
                for direction in self.isolated_approaches:
                    for index in self.direction_signal_groups.get(direction, []):
                        if index < len(chars):
                            chars[index] = "r"
                return "".join(chars)

        green_directions: Set[str] = set()
        yellow_directions: Set[str] = set()

        if self.state == STATE_GREEN:
            green_directions = self._green_directions(self.current_phase)
        elif self.state == STATE_YELLOW:
            yellow_directions = self._green_directions(self.current_phase)
        # STATE_ALL_RED: both sets stay empty -> every link is red.

        chars = []
        for index in range(self.num_links):
            direction = self._direction_for_index(index)
            if direction in green_directions:
                chars.append("G")
            elif direction in yellow_directions:
                chars.append("y")
            else:
                chars.append("r")
        return "".join(chars)


    def _direction_for_index(self, index: int) -> Optional[str]:
        return self._index_to_direction.get(index)

    def _apply_state_to_traci(self) -> None:
        state_string = self._build_state_string()
        traci = self._get_traci()
        if traci is None:
            return
        try:
            traci.trafficlight.setRedYellowGreenState(self.tls_id, state_string)
        except Exception as exc:  # pragma: no cover - defensive logging path
            print(f"[signal_controller] Failed to set traffic light state: {exc}")

    def _get_traci(self):
        if self._traci is not None:
            return self._traci
        try:
            import traci  # type: ignore

            self._traci = traci
            return traci
        except Exception:
            # No TraCI connection available (e.g. running outside SUMO for
            # unit tests) -- state is still tracked internally, it just
            # isn't pushed anywhere.
            return None