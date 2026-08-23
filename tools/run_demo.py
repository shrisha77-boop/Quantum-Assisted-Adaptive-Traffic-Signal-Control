"""
run_demo.py
===========
One-command demonstration script for the Adaptive Traffic Signal Controller.

What it does:
  1. Clears old decision results
  2. Runs the simulation headlessly with --solver adaptive (FAST, no GUI delay)
  3. After simulation finishes, generates the performance report PNG
  4. Prints a summary table

Usage:
    python tools/run_demo.py

Options:
    --interval  Decision interval in seconds (default 30)
    --solver    Solver type: classical | simulated_annealing | qaoa | adaptive (default adaptive)
    --out       Output PNG path (default analysis/performance_report.png)
    --gui       Open SUMO GUI during simulation
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "simulation" / "results" / "decision_results.jsonl"
DEFAULT_OUT  = str(ROOT / "analysis" / "performance_report_updated.png")


def run_simulation(solver: str, interval: float, use_gui: bool):
    print("\n" + "=" * 65)
    print("  STEP 1/2: Running Simulation")
    print("=" * 65)

    # Clear old results
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
        print(f"  [info] Cleared old results: {RESULTS_FILE}")

    cmd = [
        sys.executable,
        str(ROOT / "main_controller.py"),
        "--solver", solver,
        "--interval", str(interval),
        "--output-dir", str(ROOT / "simulation" / "results"),
    ]
    if use_gui:
        cmd.append("--gui")

    print(f"  [cmd]  {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print("\n[demo] Simulation exited with error. Check SUMO/TraCI setup.")
        sys.exit(result.returncode)


def run_graph(out_path: str):
    print("\n" + "=" * 65)
    print("  STEP 2/2: Generating Performance Report")
    print("=" * 65)

    if not RESULTS_FILE.exists():
        print("  [warn] No decision results found — simulation may have failed.")
        return

    cmd = [
        sys.executable,
        str(ROOT / "analysis" / "plot_comparison.py"),
        "--file", str(RESULTS_FILE),
        "--out",  out_path,
    ]
    print(f"  [cmd]  {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=str(ROOT))
    print(f"\n  [done] Report → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="One-command Adaptive Traffic Controller Demo")
    parser.add_argument("--solver",   default="adaptive",
                        choices=["simulated_annealing", "qaoa", "adaptive"])
    parser.add_argument("--interval", type=float, default=30.0,
                        help="Decision interval (seconds)")
    parser.add_argument("--out",      default=DEFAULT_OUT,
                        help="Output PNG path for performance report")
    parser.add_argument("--gui",      action="store_true",
                        help="Run with SUMO GUI")
    args = parser.parse_args()

    print("\n" + "#" * 65)
    print("#  ADAPTIVE TRAFFIC SIGNAL CONTROLLER — DEMO RUN")
    print(f"#  Solver: {args.solver}   Interval: {args.interval}s")
    print("#" * 65)

    run_simulation(args.solver, args.interval, args.gui)
    run_graph(args.out)

    print("\n" + "#" * 65)
    print("#  DEMO COMPLETE")
    print(f"#  Graph: {args.out}")
    print("#" * 65 + "\n")


if __name__ == "__main__":
    main()
