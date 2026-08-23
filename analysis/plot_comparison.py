"""
plot_comparison.py
==================
Static post-run performance analysis for the Adaptive Traffic Signal System.

Generates a rich 6-panel PNG figure from the decision_results.jsonl log:
  [1] Max Waiting Time over Time (per junction)
  [2] Queue Length over Time (per junction)
  [3] Green Time Allocated (per junction)
  [4] Peak Density over Time (per junction)
  [5] Phase Selected (NS vs EW) per Junction Heatmap
  [6] Decision Reason Breakdown (bar chart)

Usage:
    python analysis/plot_comparison.py
    python analysis/plot_comparison.py --file simulation/results/decision_results.jsonl
    python analysis/plot_comparison.py --out analysis/my_report.png
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

DEFAULT_IN  = "simulation/results/decision_results.jsonl"
DEFAULT_OUT = "analysis/performance_report_updated.png"

JUNCTION_COLORS = {
    "J1": "#e74c3c",
    "J2": "#e67e22",
    "J3": "#f1c40f",
    "J4": "#2ecc71",
    "J5": "#3498db",
}

REASON_COLORS = {
    "qubo_optimisation":     "#3498db",
    "starvation_override":   "#e74c3c",
    "high_density_override": "#e67e22",
    "emergency_preemption":  "#9b59b6",
    "isolation_constraint":  "#1abc9c",
    "other":                 "#95a5a6",
}


def load_records(path: str) -> List[dict]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def parse_records(records: List[dict]):
    times:      Dict[str, List[float]] = defaultdict(list)
    wait:       Dict[str, List[float]] = defaultdict(list)
    queue:      Dict[str, List[float]] = defaultdict(list)
    density:    Dict[str, List[float]] = defaultdict(list)
    green_time: Dict[str, List[float]] = defaultdict(list)
    phase_seq:  Dict[str, List[int]]   = defaultdict(list)   # NS=1, EW=0
    reasons:    Dict[str, int]          = defaultdict(int)
    junctions:  List[str] = []

    for r in records:
        jid = r.get("junction_id", "?")
        if jid not in junctions:
            junctions.append(jid)

        t = float(r.get("simulation_time", 0))
        times[jid].append(t)

        metrics = r.get("metrics", {})
        wt = [float(v.get("wait_time",    0) or 0) for v in metrics.values()]
        ql = [float(v.get("queue_length", 0) or 0) for v in metrics.values()]
        dn = [float(v.get("density",      0) or 0) for v in metrics.values()]

        # Use NaN when no vehicle data is present to avoid artificial zero drops in the plot
        wt_max = max(wt) if wt else np.nan
        wait[jid].append(wt_max)  # Append NaN instead of 0 when no vehicles
        queue[jid].append(sum(ql))
        density[jid].append(max(dn) if dn else 0.0)
        green_time[jid].append(float(r.get("green_time", 0)))

        phase = r.get("selected_phase", "EW")
        phase_seq[jid].append(1 if phase == "NS" else 0)

        reason = r.get("reason", "other")
        if reason not in REASON_COLORS:
            reason = "other"
        reasons[reason] += 1

    return junctions, times, wait, queue, density, green_time, phase_seq, reasons


def print_summary(junctions, wait, queue, density, green_time, reasons):
    print("\n" + "=" * 70)
    print("  ADAPTIVE TRAFFIC CONTROLLER — PERFORMANCE SUMMARY")
    print("=" * 70)
    for jid in junctions:
        w  = wait[jid]
        q  = queue[jid]
        dn = density[jid]
        gt = green_time[jid]
        print(f"\n  Junction {jid}")
        # Use nan‑aware statistics to ignore missing entries
        print(f"    Avg Wait Time  : {np.nanmean(w):.1f}s   Max: {np.nanmax(w):.1f}s")
        print(f"    Avg Queue      : {np.nanmean(q):.1f} veh  Max: {np.nanmax(q):.0f} veh")
        print(f"    Avg Density    : {np.nanmean(dn):.4f} PCU/m  Max: {np.nanmax(dn):.4f}")
        print(f"    Avg Green Time : {np.nanmean(gt):.1f}s  "
              f"Min: {np.nanmin(gt):.1f}s  Max: {np.nanmax(gt):.1f}s")

    print(f"\n  Decision Reasons:")
    total = sum(reasons.values()) or 1
    for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {r:30s}: {cnt:4d}  ({100*cnt/total:.1f}%)")
    print("=" * 70 + "\n")


def generate_chart(junctions, times, wait, queue, density, green_time,
                   phase_seq, reasons, out_path: str):

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 12), facecolor="#1a1a2e")
    fig.suptitle(
        "Adaptive Traffic Signal Controller — Performance Report",
        fontsize=17, color="white", fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           left=0.06, right=0.97,
                           top=0.93, bottom=0.07,
                           hspace=0.48, wspace=0.35)

    ax_wait    = fig.add_subplot(gs[0, 0:2])
    ax_queue   = fig.add_subplot(gs[1, 0:2])
    ax_density = fig.add_subplot(gs[2, 0:2])
    ax_green   = fig.add_subplot(gs[0, 2])
    ax_pie     = fig.add_subplot(gs[1, 2])
    ax_phase   = fig.add_subplot(gs[2, 2])

    panel_bg = "#16213e"
    all_axes = [ax_wait, ax_queue, ax_density, ax_green, ax_pie, ax_phase]
    for ax in all_axes:
        ax.set_facecolor(panel_bg)
        for spine in ax.spines.values():
            spine.set_color("#444466")
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # ── Panel 1: Waiting Time ──────────────────────────────────────────────
    ax_wait.set_title("Max Waiting Time per Junction (s)", fontsize=10, pad=6)
    ax_wait.set_xlabel("Simulation Time (s)");  ax_wait.set_ylabel("Wait Time (s)")
    for jid in junctions:
        clr = JUNCTION_COLORS.get(jid, "#fff")
        ax_wait.plot(times[jid], wait[jid], color=clr, linewidth=1.8, label=jid)
    ax_wait.axhline(y=120, color="#ff6b6b", linestyle="--", linewidth=1.0,
                    alpha=0.8, label="Starvation threshold (120s)")
    ax_wait.legend(fontsize=7, loc="upper left", framealpha=0.3)
    ax_wait.grid(axis="y", linestyle="--", alpha=0.2)

    # ── Panel 2: Queue Length ─────────────────────────────────────────────
    ax_queue.set_title("Total Queue Length per Junction (vehicles)", fontsize=10, pad=6)
    ax_queue.set_xlabel("Simulation Time (s)"); ax_queue.set_ylabel("Queue Length (veh)")
    for jid in junctions:
        clr = JUNCTION_COLORS.get(jid, "#fff")
        ax_queue.plot(times[jid], queue[jid], color=clr, linewidth=1.8, label=jid)
    ax_queue.legend(fontsize=7, loc="upper left", framealpha=0.3)
    ax_queue.grid(axis="y", linestyle="--", alpha=0.2)

    # ── Panel 3: Density ──────────────────────────────────────────────────
    ax_density.set_title("Peak Density per Junction (PCU/m)", fontsize=10, pad=6)
    ax_density.set_xlabel("Simulation Time (s)"); ax_density.set_ylabel("Density (PCU/m)")
    for jid in junctions:
        clr = JUNCTION_COLORS.get(jid, "#fff")
        ax_density.plot(times[jid], density[jid], color=clr, linewidth=1.8, label=jid)
    ax_density.axhline(y=0.65, color="#e67e22", linestyle="--", linewidth=1.0,
                       alpha=0.8, label="Density override (0.65)")
    ax_density.legend(fontsize=7, loc="upper left", framealpha=0.3)
    ax_density.grid(axis="y", linestyle="--", alpha=0.2)

    # ── Panel 4: Green Time ───────────────────────────────────────────────
    ax_green.set_title("Green Time Allocated (s)", fontsize=10, pad=6)
    ax_green.set_xlabel("Simulation Time (s)"); ax_green.set_ylabel("Green Time (s)")
    for jid in junctions:
        clr = JUNCTION_COLORS.get(jid, "#fff")
        ax_green.plot(times[jid], green_time[jid], color=clr, linewidth=1.5,
                      alpha=0.85, label=jid)
    ax_green.axhline(y=15, color="#888", linestyle=":", linewidth=0.8, alpha=0.6,
                     label="MIN (15s)")
    ax_green.axhline(y=60, color="#aaa", linestyle=":", linewidth=0.8, alpha=0.6,
                     label="MAX (60s)")
    ax_green.legend(fontsize=6, loc="upper left", framealpha=0.3)
    ax_green.grid(axis="y", linestyle="--", alpha=0.2)

    # ── Panel 5: Decision Reason Pie ──────────────────────────────────────
    ax_pie.set_title("Decision Reason Breakdown", fontsize=10, pad=6)
    reasons_nz = {k: v for k, v in reasons.items() if v > 0}
    if reasons_nz:
        labels  = [l.replace("_", "\n") for l in reasons_nz.keys()]
        sizes   = list(reasons_nz.values())
        colors  = [REASON_COLORS.get(k, "#95a5a6") for k in reasons_nz.keys()]
        wedges, texts, autotexts = ax_pie.pie(
            sizes, labels=None, colors=colors,
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 7, "color": "white"},
            wedgeprops={"linewidth": 0.8, "edgecolor": "#1a1a2e"},
            pctdistance=0.78,
        )
        for at in autotexts:
            at.set_color("white"); at.set_fontsize(7)
        ax_pie.legend(
            wedges,
            [l.replace("\n", " ") for l in labels],
            fontsize=6, loc="lower center",
            bbox_to_anchor=(0.5, -0.32), framealpha=0.3, labelcolor="white",
        )

    # ── Panel 6: Phase Heatmap ────────────────────────────────────────────
    ax_phase.set_title("Phase Selected per Junction (NS=1 / EW=0)", fontsize=10, pad=6)
    if junctions and any(phase_seq[jid] for jid in junctions):
        mat_len = max(len(phase_seq[jid]) for jid in junctions)
        mat = np.zeros((len(junctions), mat_len))
        for i, jid in enumerate(junctions):
            seq = phase_seq[jid]
            mat[i, :len(seq)] = seq

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "phase_cmap", ["#e74c3c", "#2ecc71"]
        )
        im = ax_phase.imshow(mat, aspect="auto", cmap=cmap,
                             vmin=0, vmax=1, interpolation="nearest")
        ax_phase.set_yticks(range(len(junctions)))
        ax_phase.set_yticklabels(junctions, fontsize=8, color="white")
        ax_phase.set_xlabel("Decision Cycle", color="white")
        fig.colorbar(im, ax=ax_phase, orientation="vertical",
                     fraction=0.046, pad=0.04,
                     ticks=[0, 1], format=matplotlib.ticker.FixedFormatter(["EW", "NS"]))

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n[plot_comparison] Report saved -> {os.path.abspath(out_path)}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate performance report from decision log")
    parser.add_argument("--file", default=DEFAULT_IN)
    parser.add_argument("--out",  default=DEFAULT_OUT)
    args = parser.parse_args()

    records = load_records(args.file)
    if not records:
        print(f"[plot_comparison] No records found in {args.file}. "
              "Run the simulation first: python main_controller.py --solver adaptive")
        return

    (junctions, times, wait, queue,
     density, green_time, phase_seq, reasons) = parse_records(records)

    print_summary(junctions, wait, queue, density, green_time, reasons)
    generate_chart(junctions, times, wait, queue, density,
                   green_time, phase_seq, reasons, args.out)


if __name__ == "__main__":
    main()
