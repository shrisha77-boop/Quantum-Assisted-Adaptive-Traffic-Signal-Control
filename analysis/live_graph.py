"""
live_graph.py
=============
Real-time monitoring dashboard for the Adaptive Traffic Signal Controller.

Usage (while simulation is running in another terminal):
    python analysis/live_graph.py

Or to view results after simulation:
    python analysis/live_graph.py --file simulation/results/decision_results.jsonl

Generates a 6-panel live-updating dashboard:
  [1] Avg waiting time per junction over time
  [2] Queue length per junction over time
  [3] Green time allocated per junction over time
  [4] Phase selected heatmap (NS=1, EW=0) over decision cycles
  [5] Decision reason breakdown (pie chart, live updating)
  [6] Density per junction over time
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("TkAgg")  # interactive backend; fallback to Agg if TkAgg unavailable
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import numpy as np

# ────────────────────────────────────────────────────────────────────────────
# Colour palette
# ────────────────────────────────────────────────────────────────────────────
JUNCTION_COLORS = {
    "J1": "#e74c3c",
    "J2": "#e67e22",
    "J3": "#f1c40f",
    "J4": "#2ecc71",
    "J5": "#3498db",
}
REASON_COLORS = {
    "qubo_optimisation":   "#3498db",
    "starvation_override": "#e74c3c",
    "high_density_override": "#e67e22",
    "emergency_preemption":  "#9b59b6",
    "isolation_constraint":  "#1abc9c",
    "other":               "#95a5a6",
}
PHASE_COLORS = {"NS": "#2ecc71", "EW": "#e74c3c"}

DEFAULT_FILE = "simulation/results/decision_results.jsonl"


# ────────────────────────────────────────────────────────────────────────────
# Data reader
# ────────────────────────────────────────────────────────────────────────────
class ResultsReader:
    def __init__(self, path: str):
        self.path = path
        self._last_pos = 0
        self._records: List[dict] = []

    def poll(self) -> List[dict]:
        """Return any new records since last poll (tail-follows the file)."""
        new = []
        if not os.path.exists(self.path):
            return new
        with open(self.path, "r", encoding="utf-8") as fh:
            fh.seek(self._last_pos)
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        new.append(json.loads(line))
                    except Exception:
                        pass
            self._last_pos = fh.tell()
        self._records.extend(new)
        return new

    @property
    def all_records(self):
        return self._records


# ────────────────────────────────────────────────────────────────────────────
# Data store
# ────────────────────────────────────────────────────────────────────────────
class DataStore:
    def __init__(self):
        self.times: Dict[str, List[float]] = defaultdict(list)
        self.wait: Dict[str, List[float]] = defaultdict(list)
        self.queue: Dict[str, List[float]] = defaultdict(list)
        self.density: Dict[str, List[float]] = defaultdict(list)
        self.green_time: Dict[str, List[float]] = defaultdict(list)
        self.phase: Dict[str, List[int]] = defaultdict(list)   # NS=1, EW=0
        self.reasons: Dict[str, int] = defaultdict(int)
        self.junctions: List[str] = []

    def ingest(self, record: dict):
        jid = record.get("junction_id", "?")
        if jid not in self.junctions:
            self.junctions.append(jid)

        t = float(record.get("simulation_time", 0))
        self.times[jid].append(t)

        # Aggregate metrics across all directions
        metrics = record.get("metrics", {})
        wt = [float(v.get("wait_time", 0) or 0) for v in metrics.values()]
        ql = [float(v.get("queue_length", 0) or 0) for v in metrics.values()]
        dn = [float(v.get("density", 0) or 0) for v in metrics.values()]

        self.wait[jid].append(max(wt) if wt else 0.0)
        self.queue[jid].append(sum(ql))
        self.density[jid].append(max(dn) if dn else 0.0)
        self.green_time[jid].append(float(record.get("green_time", 0)))

        phase = record.get("selected_phase", "EW")
        self.phase[jid].append(1 if phase == "NS" else 0)

        reason = record.get("reason", "other")
        if reason not in REASON_COLORS:
            reason = "other"
        self.reasons[reason] += 1


# ────────────────────────────────────────────────────────────────────────────
# Dashboard
# ────────────────────────────────────────────────────────────────────────────
class Dashboard:
    def __init__(self, reader: ResultsReader, store: DataStore, interval_ms: int = 1000):
        self.reader = reader
        self.store = store
        self.interval_ms = interval_ms

        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(18, 10), facecolor="#1a1a2e")
        self.fig.suptitle(
            "Adaptive Traffic Signal Controller — Live Dashboard",
            fontsize=15, color="white", fontweight="bold", y=0.98
        )

        gs = gridspec.GridSpec(3, 3, figure=self.fig,
                               left=0.06, right=0.97,
                               top=0.93, bottom=0.07,
                               hspace=0.45, wspace=0.38)

        self.ax_wait    = self.fig.add_subplot(gs[0, 0:2])
        self.ax_queue   = self.fig.add_subplot(gs[1, 0:2])
        self.ax_density = self.fig.add_subplot(gs[2, 0:2])
        self.ax_green   = self.fig.add_subplot(gs[0, 2])
        self.ax_pie     = self.fig.add_subplot(gs[1, 2])
        self.ax_phase   = self.fig.add_subplot(gs[2, 2])

        self._style_axes()

    def _style_axes(self):
        panel_bg = "#16213e"
        for ax in [self.ax_wait, self.ax_queue, self.ax_density,
                   self.ax_green, self.ax_pie, self.ax_phase]:
            ax.set_facecolor(panel_bg)
            for spine in ax.spines.values():
                spine.set_color("#444466")
            ax.tick_params(colors="white", labelsize=7)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

        self.ax_wait.set_title("Max Waiting Time per Junction (s)", fontsize=9)
        self.ax_wait.set_xlabel("Simulation Time (s)")
        self.ax_wait.set_ylabel("Wait Time (s)")

        self.ax_queue.set_title("Total Queue Length per Junction (vehicles)", fontsize=9)
        self.ax_queue.set_xlabel("Simulation Time (s)")
        self.ax_queue.set_ylabel("Queue Length")

        self.ax_density.set_title("Peak Density per Junction (PCU/m)", fontsize=9)
        self.ax_density.set_xlabel("Simulation Time (s)")
        self.ax_density.set_ylabel("Density")

        self.ax_green.set_title("Green Time Allocated (s)", fontsize=9)
        self.ax_green.set_xlabel("Simulation Time (s)")
        self.ax_green.set_ylabel("Green Time (s)")

        self.ax_pie.set_title("Decision Reason Breakdown", fontsize=9)

        self.ax_phase.set_title("Phase Selected (NS=1 / EW=0)", fontsize=9)
        self.ax_phase.set_xlabel("Decision Cycle")
        self.ax_phase.set_ylabel("Phase")

    def _update(self, _frame):
        self.reader.poll()

        store = self.store
        junctions = store.junctions or []

        # Clear line plots
        for ax in [self.ax_wait, self.ax_queue, self.ax_density,
                   self.ax_green, self.ax_phase]:
            ax.cla()
        self._style_axes()

        for jid in junctions:
            clr = JUNCTION_COLORS.get(jid, "#ffffff")
            times = store.times[jid]
            if not times:
                continue

            self.ax_wait.plot(times, store.wait[jid],   color=clr, linewidth=1.5, label=jid)
            self.ax_queue.plot(times, store.queue[jid],  color=clr, linewidth=1.5, label=jid)
            self.ax_density.plot(times, store.density[jid], color=clr, linewidth=1.5, label=jid)
            self.ax_green.plot(times, store.green_time[jid], color=clr, linewidth=1.5, label=jid)

            # Phase step chart
            cycles = list(range(len(store.phase[jid])))
            self.ax_phase.step(cycles, store.phase[jid], where="post",
                               color=clr, linewidth=1, alpha=0.8, label=jid)

        # Starvation threshold line
        self.ax_wait.axhline(y=120, color="#ff6b6b", linestyle="--",
                             linewidth=0.8, alpha=0.7, label="Starvation (120s)")
        self.ax_density.axhline(y=0.65, color="#ff6b6b", linestyle="--",
                                linewidth=0.8, alpha=0.7, label="Density override (0.65)")

        for ax in [self.ax_wait, self.ax_queue, self.ax_density, self.ax_green]:
            if junctions:
                ax.legend(fontsize=6, loc="upper left", framealpha=0.3)

        # Pie chart
        self.ax_pie.cla()
        self.ax_pie.set_facecolor("#16213e")
        self.ax_pie.title.set_text("Decision Reason Breakdown")
        self.ax_pie.title.set_color("white")
        reasons = {k: v for k, v in store.reasons.items() if v > 0}
        if reasons:
            labels = list(reasons.keys())
            sizes  = list(reasons.values())
            colors = [REASON_COLORS.get(r, "#95a5a6") for r in labels]
            wedges, texts, autotexts = self.ax_pie.pie(
                sizes, labels=None, colors=colors,
                autopct="%1.0f%%", startangle=90,
                textprops={"fontsize": 7, "color": "white"},
                wedgeprops={"linewidth": 0.5, "edgecolor": "#1a1a2e"},
            )
            for at in autotexts:
                at.set_color("white")
                at.set_fontsize(7)
            self.ax_pie.legend(
                wedges, [l.replace("_", " ") for l in labels],
                fontsize=6, loc="lower center",
                bbox_to_anchor=(0.5, -0.25),
                framealpha=0.3, labelcolor="white",
            )
        else:
            self.ax_pie.text(0.5, 0.5, "No decisions yet…",
                             transform=self.ax_pie.transAxes,
                             ha="center", va="center", color="white", fontsize=9)

        self.fig.canvas.draw_idle()

    def run(self):
        ani = animation.FuncAnimation(
            self.fig, self._update, interval=self.interval_ms, cache_frame_data=False
        )
        plt.show()


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Live traffic signal monitoring dashboard")
    parser.add_argument("--file", default=DEFAULT_FILE,
                        help=f"Path to decision_results.jsonl (default: {DEFAULT_FILE})")
    parser.add_argument("--interval", type=int, default=1000,
                        help="Refresh interval in ms (default 1000)")
    args = parser.parse_args()

    reader = ResultsReader(args.file)
    store  = DataStore()

    # Pre-load existing data
    for rec in reader.poll():
        store.ingest(rec)

    # Register reader → store ingestion as the poll hook
    original_poll = reader.poll
    def _poll_and_ingest():
        for rec in original_poll():
            store.ingest(rec)
        return []
    reader.poll = _poll_and_ingest
    # Restore reader.all_records access
    reader._records = store.wait  # dummy ref (not used after this)

    # Rebuild store from already-loaded records
    reader._last_pos = 0
    reader._records  = []
    store2 = DataStore()
    for rec in ResultsReader(args.file).poll():
        store2.ingest(rec)

    reader2 = ResultsReader(args.file)
    for rec in reader2.poll():
        store2.ingest(rec)

    def _update_store(frame):
        if not os.path.exists(args.file):
            return
        with open(args.file, "r", encoding="utf-8") as fh:
            fh.seek(reader2._last_pos)
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        store2.ingest(json.loads(line))
                    except Exception:
                        pass
            reader2._last_pos = fh.tell()

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 10), facecolor="#1a1a2e")
    fig.suptitle(
        "Adaptive Traffic Signal Controller — Live Dashboard",
        fontsize=15, color="white", fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           left=0.06, right=0.97,
                           top=0.93, bottom=0.07,
                           hspace=0.45, wspace=0.38)

    ax_wait    = fig.add_subplot(gs[0, 0:2])
    ax_queue   = fig.add_subplot(gs[1, 0:2])
    ax_density = fig.add_subplot(gs[2, 0:2])
    ax_green   = fig.add_subplot(gs[0, 2])
    ax_pie     = fig.add_subplot(gs[1, 2])
    ax_phase   = fig.add_subplot(gs[2, 2])

    panel_bg = "#16213e"
    all_axes = [ax_wait, ax_queue, ax_density, ax_green, ax_pie, ax_phase]

    def style_all():
        for ax in all_axes:
            ax.set_facecolor(panel_bg)
            for spine in ax.spines.values():
                spine.set_color("#444466")
            ax.tick_params(colors="white", labelsize=7)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

    def animate(frame):
        _update_store(frame)
        store = store2
        junctions = store.junctions

        for ax in [ax_wait, ax_queue, ax_density, ax_green, ax_phase]:
            ax.cla()
        style_all()

        ax_wait.set_title("Max Waiting Time per Junction (s)", fontsize=9)
        ax_wait.set_xlabel("Simulation Time (s)"); ax_wait.set_ylabel("Wait Time (s)")
        ax_queue.set_title("Total Queue Length per Junction (vehicles)", fontsize=9)
        ax_queue.set_xlabel("Simulation Time (s)"); ax_queue.set_ylabel("Queue Length")
        ax_density.set_title("Peak Density per Junction (PCU/m)", fontsize=9)
        ax_density.set_xlabel("Simulation Time (s)"); ax_density.set_ylabel("Density")
        ax_green.set_title("Green Time Allocated (s)", fontsize=9)
        ax_green.set_xlabel("Simulation Time (s)"); ax_green.set_ylabel("Green Time (s)")
        ax_phase.set_title("Phase Selected  (NS=green / EW=red)", fontsize=9)
        ax_phase.set_xlabel("Decision Cycle"); ax_phase.set_ylabel("Phase (1=NS, 0=EW)")

        for jid in junctions:
            clr = JUNCTION_COLORS.get(jid, "#ffffff")
            times = store.times[jid]
            if not times:
                continue
            ax_wait.plot(times, store.wait[jid],     color=clr, linewidth=1.5, label=jid)
            ax_queue.plot(times, store.queue[jid],   color=clr, linewidth=1.5, label=jid)
            ax_density.plot(times, store.density[jid], color=clr, linewidth=1.5, label=jid)
            ax_green.plot(times, store.green_time[jid], color=clr, linewidth=1.5, label=jid)
            cycles = list(range(len(store.phase[jid])))
            ax_phase.step(cycles, store.phase[jid], where="post",
                          color=clr, linewidth=1.2, alpha=0.85, label=jid)

        ax_wait.axhline(y=120, color="#ff6b6b", linestyle="--",
                        linewidth=0.8, alpha=0.7, label="Starvation (120s)")
        ax_density.axhline(y=0.65, color="#ff6b6b", linestyle="--",
                           linewidth=0.8, alpha=0.7, label="Density (0.65)")

        for ax in [ax_wait, ax_queue, ax_density, ax_green]:
            if junctions:
                ax.legend(fontsize=6, loc="upper left", framealpha=0.3)
        if junctions:
            ax_phase.legend(fontsize=6, loc="upper right", framealpha=0.3)

        # Pie chart
        ax_pie.cla()
        ax_pie.set_facecolor(panel_bg)
        ax_pie.set_title("Decision Reason Breakdown", fontsize=9, color="white")
        reasons = {k: v for k, v in store.reasons.items() if v > 0}
        if reasons:
            labels = list(reasons.keys())
            sizes  = list(reasons.values())
            colors = [REASON_COLORS.get(r, "#95a5a6") for r in labels]
            wedges, texts, autotexts = ax_pie.pie(
                sizes, labels=None, colors=colors,
                autopct="%1.0f%%", startangle=90,
                textprops={"fontsize": 7, "color": "white"},
                wedgeprops={"linewidth": 0.5, "edgecolor": "#1a1a2e"},
            )
            for at in autotexts:
                at.set_color("white"); at.set_fontsize(7)
            ax_pie.legend(
                wedges, [l.replace("_", " ") for l in labels],
                fontsize=6, loc="lower center",
                bbox_to_anchor=(0.5, -0.3),
                framealpha=0.3, labelcolor="white",
            )
        else:
            ax_pie.text(0.5, 0.5, "No decisions yet…",
                        transform=ax_pie.transAxes,
                        ha="center", va="center", color="white", fontsize=9)

    style_all()
    ani = animation.FuncAnimation(fig, animate, interval=args.interval, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
