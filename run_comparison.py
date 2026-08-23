"""
run_comparison.py
=================
Automated benchmarking and comparison tool for the Bengaluru Adaptive Traffic Signal System.

Executes:
  1. SUMO Baseline (Default fixed-time actuated controller)
  2. Proposed Adaptive Method (Quantum QAOA / Simulated Annealing with Emergency Preemption,
     Road Isolation, and Predictor)

Generates:
  - analysis/comparison_report.png (Comprehensive 6-panel graphical comparison)
  - Console summary table comparing Waiting Time, Time Loss, Throughput, and Emergency Speeds.

Usage:
  python run_comparison.py
  python run_comparison.py --duration 600
  python run_comparison.py --duration 1200 --gui
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from scenario import scenario_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_comparison")


def parse_tripinfo(xml_path: Path) -> List[Dict[str, Any]]:
    """Parse SUMO tripinfo.xml for completed trip metrics."""
    if not xml_path.exists():
        return []
    trips = []
    try:
        tree = ET.parse(xml_path)
        for t in tree.getroot().findall("tripinfo"):
            trips.append({
                "id": t.get("id"),
                "vType": t.get("vType"),
                "depart": float(t.get("depart", 0.0)),
                "arrival": float(t.get("arrival", 0.0)),
                "duration": float(t.get("duration", 0.0)),
                "waitingTime": float(t.get("waitingTime", 0.0)),
                "timeLoss": float(t.get("timeLoss", 0.0)),
            })
    except Exception as exc:
        logger.error("Error reading %s: %s", xml_path, exc)
    return trips


def run_baseline_simulation(duration: int, output_dir: Path, use_gui: bool = False) -> Path:
    """Run SUMO baseline (unmanaged default TLS programs)."""
    tripinfo_path = output_dir / "tripinfo_baseline.xml"
    sumo_bin = "sumo-gui" if use_gui else "sumo"
    cmd = [
        sumo_bin,
        "-c", scenario_config.SUMO_CONFIG,
        "--step-length", str(config.SUMO_STEP_LENGTH),
        "--no-step-log", "true",
        "--duration-log.disable", "true",
        "--end", str(duration),
        "--tripinfo-output", str(tripinfo_path),
        "--tripinfo-output.write-unfinished", "true",  # include ambulances still en route at sim end
    ]
    if use_gui:
        cmd.append("--start")
    logger.info(">>> Running Baseline SUMO Simulation (%ds)...", duration)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    logger.info("Baseline finished in %.2fs. Tripinfo saved to %s", time.time() - t0, tripinfo_path)
    return tripinfo_path


def run_adaptive_simulation(duration: int, output_dir: Path, solver: str = "adaptive", use_gui: bool = False) -> Path:
    """Run Proposed Adaptive Controller."""
    import main_controller

    tripinfo_path = output_dir / "tripinfo_adaptive.xml"
    logger.info(">>> Running Proposed Adaptive (%s) Simulation (%ds)...", solver, duration)
    t0 = time.time()

    # Pass tripinfo output argument via extra_sumo_args in TraCIInterface
    from modules.traci_interface import TraCIConfig, TraCIInterface
    from modules.data_collection import DataCollectionLayer

    mc = main_controller.MainController(
        sumo_config=scenario_config.SUMO_CONFIG,
        use_gui=use_gui,
        decision_interval=30.0,
        solver=solver,
        output_dir=str(output_dir),
    )
    # Configure tripinfo output
    mc.traci_api = TraCIInterface(
        TraCIConfig(
            sumocfg_path=mc.sumo_config,
            use_gui=mc.use_gui,
            step_length=config.SUMO_STEP_LENGTH,
            extra_sumo_args=[
                "--tripinfo-output", str(tripinfo_path),
                "--end", str(duration),
                "--tripinfo-output.write-unfinished", "true",  # include ambulances still en route at sim end
            ],
            collect_edges=False,
            collect_junctions=False,
            collect_traffic_lights=False,
            collect_turning_movements=False,
        )
    )
    mc.data_collection = DataCollectionLayer()
    mc.traci_api.start_simulation()
    mc.traci = __import__("traci")
    mc._initialize_runtime_modules()

    try:
        sim_time = mc.traci_api.step()
        snapshot = mc._collect_fast_snapshot()
        mc.data_collection.update(snapshot)

        while True:
            sim_time_f = float(sim_time) if sim_time is not None else 0.0
            if sim_time_f >= duration or not mc.traci_api.is_running():
                break

            # Automated incident demonstration (t=150s to 450s)
            mc._handle_incident_injection(sim_time_f)

            if sim_time_f >= mc.next_decision_time and not mc._decisions_in_flight:
                mc._decisions_in_flight = True
                # Run synchronously for benchmark determinism
                mc._run_decision_cycle(sim_time_f)
                mc.decision_count += 1
                mc.next_decision_time = sim_time_f + mc.decision_interval

            with mc._decision_lock:
                current_decisions = dict(mc.latest_decisions)

            for junction_id, controller in mc.signal_controllers.items():
                decision = current_decisions.get(junction_id)
                if decision is None:
                    continue
                force_switch = (decision.emergency or decision.starvation_override) and (
                    controller.current_phase != decision.phase
                    and controller.pending_phase != decision.phase
                )
                controller.apply(
                    decision.phase,
                    decision.green_time,
                    isolated_approaches=decision.isolated_approaches,
                    force=force_switch,
                )

            sim_time = mc.traci_api.step()
            snapshot = mc._collect_fast_snapshot()
            mc.data_collection.update(snapshot)
    finally:
        mc.shutdown()

    logger.info("Proposed Adaptive finished in %.2fs. Tripinfo saved to %s", time.time() - t0, tripinfo_path)
    return tripinfo_path


def generate_comparison_plot(
    baseline_trips: List[Dict[str, Any]],
    adaptive_trips: List[Dict[str, Any]],
    decision_records: List[Dict[str, Any]],
    output_png: Path,
):
    """Generate 6-panel comprehensive comparison figure."""
    output_png.parent.mkdir(parents=True, exist_ok=True)

    base_waits = [t["waitingTime"] for t in baseline_trips] or [0]
    adap_waits = [t["waitingTime"] for t in adaptive_trips] or [0]

    base_losses = [t["timeLoss"] for t in baseline_trips] or [0]
    adap_losses = [t["timeLoss"] for t in adaptive_trips] or [0]

    base_durations = [t["duration"] for t in baseline_trips] or [0]
    adap_durations = [t["duration"] for t in adaptive_trips] or [0]

    # Emergency vehicles
    base_amb = [t["duration"] for t in baseline_trips if "ambulance" in str(t.get("vType", "")).lower() or "ambulance" in str(t.get("id", "")).lower()]
    adap_amb = [t["duration"] for t in adaptive_trips if "ambulance" in str(t.get("vType", "")).lower() or "ambulance" in str(t.get("id", "")).lower()]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Adaptive Traffic Management (Quantum QAOA / SA) vs. SUMO Default", fontsize=16, fontweight="bold")

    # [1] Average Waiting Time
    ax = axes[0, 0]
    labels = ["SUMO Default", "Adaptive (Ours)"]
    means = [np.mean(base_waits), np.mean(adap_waits)]
    colors = ["#e74c3c", "#2ecc71"]
    bars = ax.bar(labels, means, color=colors, width=0.5, edgecolor="black")
    ax.set_title("Average Waiting Time per Vehicle (s)", fontweight="bold")
    ax.set_ylabel("Seconds")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}s", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # [2] Time Loss / Travel Delay
    ax = axes[0, 1]
    means_loss = [np.mean(base_losses), np.mean(adap_losses)]
    bars = ax.bar(labels, means_loss, color=["#e67e22", "#3498db"], width=0.5, edgecolor="black")
    ax.set_title("Average Travel Delay / Time Loss (s)", fontweight="bold")
    ax.set_ylabel("Seconds")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}s", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # [3] Completed Vehicle Throughput
    ax = axes[0, 2]
    counts = [len(baseline_trips), len(adaptive_trips)]
    bars = ax.bar(labels, counts, color=["#95a5a6", "#27ae60"], width=0.5, edgecolor="black")
    ax.set_title("Completed Vehicle Throughput", fontweight="bold")
    ax.set_ylabel("Vehicles Reached Destination")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{int(h)} vehs", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # [4] Emergency Vehicle Trip Duration
    ax = axes[1, 0]
    amb_labels = ["SUMO Default", "Adaptive (Preemption)"]
    amb_means = [np.mean(base_amb) if base_amb else 0.0, np.mean(adap_amb) if adap_amb else 0.0]
    bars = ax.bar(amb_labels, amb_means, color=["#e74c3c", "#9b59b6"], width=0.5, edgecolor="black")
    ax.set_title("Ambulance Travel Time (Emergency Clearance)", fontweight="bold")
    ax.set_ylabel("Seconds")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}s", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # [5] Waiting Time Progression over Decision Cycles
    ax = axes[1, 1]
    if decision_records:
        sim_times = sorted(list(set(float(r.get("simulation_time", 0)) for r in decision_records)))
        max_waits_per_time = []
        for st in sim_times:
            recs = [r for r in decision_records if float(r.get("simulation_time", 0)) == st]
            waits = []
            for r in recs:
                for m in r.get("metrics", {}).values():
                    waits.append(float(m.get("wait_time", 0) or 0))
            max_waits_per_time.append(max(waits) if waits else 0.0)
        ax.plot(sim_times, max_waits_per_time, marker="o", color="#3498db", label="Max Wait Time (Adaptive)")
        ax.axhline(y=config.MAX_WAIT_SECONDS, color="red", linestyle="--", label="Starvation Threshold (120s)")
        ax.set_title("Max Approach Waiting Time vs. Time", fontweight="bold")
        ax.set_xlabel("Simulation Time (s)")
        ax.set_ylabel("Max Wait (s)")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
    else:
        ax.text(0.5, 0.5, "No decision log records", ha="center", va="center")

    # [6] Decision Reason Breakdown
    ax = axes[1, 2]
    reasons = {}
    for r in decision_records:
        reason = r.get("reason", "other")
        reasons[reason] = reasons.get(reason, 0) + 1
    if reasons:
        labels_r = list(reasons.keys())
        sizes_r = list(reasons.values())
        colors_map = {
            "qubo_optimisation": "#3498db",
            "starvation_override": "#e74c3c",
            "high_density_override": "#e67e22",
            "emergency_preemption": "#9b59b6",
            "all_approaches_isolated_fallback": "#1abc9c",
        }
        pie_colors = [colors_map.get(k, "#95a5a6") for k in labels_r]
        ax.pie(sizes_r, labels=labels_r, autopct="%1.1f%%", colors=pie_colors, startangle=140)
        ax.set_title("Controller Decision Reason Distribution", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No decisions recorded", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    logger.info("Comparison figure generated: %s", output_png)


def print_summary_table(baseline_trips: List[Dict[str, Any]], adaptive_trips: List[Dict[str, Any]]):
    """Print comparative metrics table."""
    base_waits = [t["waitingTime"] for t in baseline_trips] or [0]
    adap_waits = [t["waitingTime"] for t in adaptive_trips] or [0]

    base_losses = [t["timeLoss"] for t in baseline_trips] or [0]
    adap_losses = [t["timeLoss"] for t in adaptive_trips] or [0]

    base_dur = [t["duration"] for t in baseline_trips] or [0]
    adap_dur = [t["duration"] for t in adaptive_trips] or [0]

    base_amb = [t["duration"] for t in baseline_trips if "ambulance" in str(t.get("vType", "")).lower() or "ambulance" in str(t.get("id", "")).lower()]
    adap_amb = [t["duration"] for t in adaptive_trips if "ambulance" in str(t.get("vType", "")).lower() or "ambulance" in str(t.get("id", "")).lower()]

    bw, aw = np.mean(base_waits), np.mean(adap_waits)
    bl, al = np.mean(base_losses), np.mean(adap_losses)
    bd, ad = np.mean(base_dur), np.mean(adap_dur)
    bt, at = len(baseline_trips), len(adaptive_trips)
    bam, aam = (np.mean(base_amb) if base_amb else 0.0), (np.mean(adap_amb) if adap_amb else 0.0)

    wait_diff = ((aw - bw) / bw * 100.0) if bw > 0 else 0.0
    loss_diff = ((al - bl) / bl * 100.0) if bl > 0 else 0.0
    dur_diff = ((ad - bd) / bd * 100.0) if bd > 0 else 0.0
    thru_diff = ((at - bt) / bt * 100.0) if bt > 0 else 0.0

    print("\n" + "=" * 80)
    print("           BENGALURU TRAFFIC MANAGEMENT: SYSTEM COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<35} | {'SUMO Default':<15} | {'Proposed Adaptive':<18} | {'Improvement'}")
    print("-" * 80)
    print(f"{'Avg Waiting Time (s/veh)':<35} | {bw:<15.2f} | {aw:<18.2f} | {wait_diff:+.1f}%")
    print(f"{'Avg Travel Delay / Time Loss (s)':<35} | {bl:<15.2f} | {al:<18.2f} | {loss_diff:+.1f}%")
    print(f"{'Avg Trip Duration (s)':<35} | {bd:<15.2f} | {ad:<18.2f} | {dur_diff:+.1f}%")
    print(f"{'Completed Vehicle Throughput':<35} | {bt:<15} | {at:<18} | {thru_diff:+.1f}%")
    if base_amb or adap_amb:
        print(f"{'Ambulance Trip Time (s)':<35} | {bam:<15.2f} | {aam:<18.2f} | {((aam-bam)/bam*100.0 if bam>0 else 0):+.1f}%")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark & Compare SUMO Baseline vs Proposed Adaptive Controller")
    parser.add_argument("--duration", type=int, default=600, help="Simulation duration in seconds (default: 600)")
    parser.add_argument("--solver", choices=["adaptive", "simulated_annealing", "qaoa"], default="adaptive")
    parser.add_argument("--gui", action="store_true", help="Launch GUI during simulation")
    parser.add_argument("--output-dir", default="simulation/results", help="Directory for result logs")
    parser.add_argument("--report-png", default="analysis/comparison_report.png", help="Path for comparison report PNG")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run SUMO baseline
    baseline_xml = run_baseline_simulation(args.duration, out_dir, use_gui=args.gui)

    # 2. Run Proposed Adaptive
    adaptive_xml = run_adaptive_simulation(args.duration, out_dir, solver=args.solver, use_gui=args.gui)

    # 3. Parse trip statistics
    baseline_trips = parse_tripinfo(baseline_xml)
    adaptive_trips = parse_tripinfo(adaptive_xml)

    # 4. Load decision records
    dec_file = out_dir / "decision_results.jsonl"
    decision_records = []
    if dec_file.exists():
        with open(dec_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        decision_records.append(json.loads(line))
                    except Exception:
                        pass

    # 5. Print summary table
    print_summary_table(baseline_trips, adaptive_trips)

    # 6. Generate graphical report
    generate_comparison_plot(baseline_trips, adaptive_trips, decision_records, Path(args.report_png))
    print(f"Report image saved to: {args.report_png}")


if __name__ == "__main__":
    main()
