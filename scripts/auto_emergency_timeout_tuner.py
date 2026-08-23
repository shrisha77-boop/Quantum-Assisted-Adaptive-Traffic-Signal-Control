#!/usr/bin/env python
# ------------------------------------------------------------
# scripts/auto_emergency_timeout_tuner.py
# ------------------------------------------------------------
"""
Automatically search for the best EMERGENCY_HOLD_TIMEOUT.

The script runs the traffic‑controller for a series of timeout
candidates, measures the average waiting time (or any KPI you prefer)
and selects the value that gives the lowest metric.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
from typing import List, Tuple

# ----------------------------------------------------------------
# Helper – run one simulation with a given timeout.
# Returns (average_wait, max_wait) across the whole simulation.
# ----------------------------------------------------------------
def run_one(timeout: int, cfg_path: pathlib.Path, sim_timeout: int) -> Tuple[float, float]:
    # The config file is assumed already edited for this timeout.
    # 1️⃣ Execute the controller (adaptive mode – the normal runtime behaviour)
    cmd = [
        sys.executable,
        "main_controller.py",
        "--solver",
        "adaptive",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(cfg_path.parent.parent),  # project root (contains main_controller.py)
        capture_output=True,
        text=True,
        timeout=sim_timeout,  # allow longer simulations
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Simulation failed (timeout={timeout})\n{proc.stderr}")

    # 2️⃣ Parse the decision_results.jsonl that the controller produced
    results_path = pathlib.Path("simulation/results/decision_results.jsonl")
    if not results_path.is_file():
        raise FileNotFoundError("decision_results.jsonl not produced by the run")

    wait_vals: List[float] = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                # maximum wait among the four approaches for this step
                w = max(v.get("wait_time", 0) for v in rec.get("metrics", {}).values())
                wait_vals.append(w)
            except json.JSONDecodeError:
                continue

    avg_wait = sum(wait_vals) / len(wait_vals) if wait_vals else float("inf")
    max_wait = max(wait_vals) if wait_vals else float("inf")
    return avg_wait, max_wait

# ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Auto‑tune EMERGENCY_HOLD_TIMEOUT")
    parser.add_argument("--min", type=int, default=5, help="minimum timeout (seconds)")
    parser.add_argument("--max", type=int, default=20, help="maximum timeout (seconds)")
    parser.add_argument("--step", type=int, default=1, help="step size (seconds)")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="path to config.py to overwrite with the optimal value (optional)",
    )
    parser.add_argument(
        "--sim-timeout",
        type=int,
        default=600,
        help="maximum seconds allowed for a single simulation run",
    )
    args = parser.parse_args()

    # Prefer the project‑root config.py; fall back to the legacy location.
    cfg_path = pathlib.Path("config.py")
    if not cfg_path.is_file():
        cfg_path = pathlib.Path("modules/qubo_solver/config.py")
    if not cfg_path.is_file():
        print("❌ config.py not found – aborting.", file=sys.stderr)
        sys.exit(1)

    # Keep a pristine copy of the original config to restore after each run
    original_src = cfg_path.read_text(encoding="utf-8")

    best_timeout = None
    best_avg = float("inf")
    results: List[Tuple[int, float, float]] = []

    print("[INFO] Searching for the optimal EMERGENCY_HOLD_TIMEOUT ...")
    for t in range(args.min, args.max + 1, args.step):
        try:
            # Edit config for this timeout
            new_src = re.sub(r"EMERGENCY_HOLD_TIMEOUT\s*=\s*\d+", f"EMERGENCY_HOLD_TIMEOUT = {t}", original_src)
            cfg_path.write_text(new_src, encoding="utf-8")
            # Run simulation
            avg, mx = run_one(t, cfg_path, args.sim_timeout)
            results.append((t, avg, mx))
            print(f"   [TIMEOUT] timeout={t:2d}s avg_wait={avg:6.1f}s max_wait={mx:6.1f}s")
            if avg < best_avg:
                best_avg = avg
                best_timeout = t
        except Exception as e:
            print(f"   ⚠️  timeout={t}s failed: {e}", file=sys.stderr)
        finally:
            # Restore the original config regardless of success/failure
            cfg_path.write_text(original_src, encoding="utf-8")

    if best_timeout is None:
        print("\n❌ No successful run – cannot suggest a timeout.", file=sys.stderr)
        sys.exit(1)

    print("\n🏁 Best timeout found:")
    print(f"   → {best_timeout}s   (average wait = {best_avg:.1f}s)")

    # Optional – write the chosen value back into the real config file
    if args.out:
        final_src = re.sub(r"EMERGENCY_HOLD_TIMEOUT\s*=\s*\d+", f"EMERGENCY_HOLD_TIMEOUT = {best_timeout}", original_src)
        args.out.write_text(final_src, encoding="utf-8")
        print(f"✅ Updated {args.out} with EMERGENCY_HOLD_TIMEOUT = {best_timeout}")

    # Summary table
    print("\n📊 All tested values")
    print(" timeout | avg_wait (s) | max_wait (s)")
    print("--------+--------------+--------------")
    for t, avg, mx in results:
        marker = "← best" if t == best_timeout else ""
        print(f"   {t:2d}   |    {avg:8.1f}   |    {mx:8.1f} {marker}")

if __name__ == "__main__":
    main()

# ------------------------------------------------------------
# scripts/auto_emergency_timeout_tuner.py
# ------------------------------------------------------------
"""
Automatically search for the best EMERGENCY_HOLD_TIMEOUT.

The script runs the traffic‑controller for a series of timeout
candidates, measures the average waiting time (or any KPI you prefer)
and selects the value that gives the lowest metric.

Usage example:
    python scripts/auto_emergency_timeout_tuner.py \
        --min 5 --max 20 --step 1 \
        --out modules/qubo_solver/config.py   # overwrite the real config
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
from typing import List, Tuple

# ----------------------------------------------------------------
# Helper – run one simulation with a given timeout.
# Returns (average_wait, max_wait) across the whole simulation.
# ----------------------------------------------------------------
def run_one(timeout: int, base_cfg: pathlib.Path) -> Tuple[float, float]:
    # 1️⃣ Edit the root config.py in-place for the given timeout.
    cfg_path = base_cfg
    src = cfg_path.read_text(encoding="utf-8")
    new_src = re.sub(r"EMERGENCY_HOLD_TIMEOUT\s*=\s*\d+", f"EMERGENCY_HOLD_TIMEOUT = {timeout}", src)
    cfg_path.write_text(new_src, encoding="utf-8")

    # 2️⃣ Execute the controller (adaptive mode – the normal runtime behaviour)
    cmd = [
        sys.executable,
        "main_controller.py",
        "--solver",
        "adaptive",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(base_cfg.parent.parent),  # project root (contains main_controller.py)
        capture_output=True,
        text=True,
        timeout=args.sim_timeout,                     # increased timeout for simulation runs
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Simulation failed (timeout={timeout})\n{proc.stderr}")

    # 3️⃣ Parse the decision_results.jsonl that the controller produced
    results_path = pathlib.Path("simulation/results/decision_results.jsonl")
    if not results_path.is_file():
        raise FileNotFoundError("decision_results.jsonl not produced by the run")

    wait_vals: List[float] = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                # maximum wait among the four approaches for this step
                w = max(v.get("wait_time", 0) for v in rec.get("metrics", {}).values())
                wait_vals.append(w)
            except json.JSONDecodeError:
                continue

    avg_wait = sum(wait_vals) / len(wait_vals) if wait_vals else float("inf")
    max_wait = max(wait_vals) if wait_vals else float("inf")
    return avg_wait, max_wait


# ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Auto‑tune EMERGENCY_HOLD_TIMEOUT")
    parser.add_argument("--min", type=int, default=5, help="minimum timeout (seconds)")
    parser.add_argument("--max", type=int, default=20, help="maximum timeout (seconds)")
    parser.add_argument("--step", type=int, default=1, help="step size (seconds)")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="path to config.py to overwrite with the optimal value (optional)",
    )
    args = parser.parse_args()

    # Prefer the project‑root config.py; fall back to the legacy location.
    cfg_path = pathlib.Path("config.py")
    if not cfg_path.is_file():
        # fallback to the old location for backward compatibility
        cfg_path = pathlib.Path("modules/qubo_solver/config.py")
    if not cfg_path.is_file():
        print("❌ config.py not found – aborting.", file=sys.stderr)
        sys.exit(1)

    best_timeout = None
    best_avg = float("inf")
    results: List[Tuple[int, float, float]] = []

    print("[INFO] Searching for the optimal EMERGENCY_HOLD_TIMEOUT ...")
    for t in range(args.min, args.max + 1, args.step):
        try:
            avg, mx = run_one(t, cfg_path)
            results.append((t, avg, mx))
            print(f"   [TIMEOUT] timeout={t:2d}s avg_wait={avg:6.1f}s max_wait={mx:6.1f}s")
            if avg < best_avg:
                best_avg = avg
                best_timeout = t
        except Exception as e:
            print(f"   ⚠️  timeout={t}s failed: {e}", file=sys.stderr)

    if best_timeout is None:
        print("\n❌ No successful run – cannot suggest a timeout.", file=sys.stderr)
        sys.exit(1)

    print("\n🏁 Best timeout found:")
    print(f"   → {best_timeout}s   (average wait = {best_avg:.1f}s)")

    # Optional – write the chosen value back into the real config file
    if args.out:
        real_src = cfg_path.read_text(encoding="utf-8")
        new_real = re.sub(r"EMERGENCY_HOLD_TIMEOUT\s*=\s*\d+", f"EMERGENCY_HOLD_TIMEOUT = {best_timeout}", real_src)
        args.out.write_text(new_real, encoding="utf-8")
        print(f"✅ Updated {args.out} with EMERGENCY_HOLD_TIMEOUT = {best_timeout}")

    # Print a final table for quick reference
    print("\n📊 All tested values")
    print(" timeout | avg_wait (s) | max_wait (s)")
    print("--------+--------------+--------------")
    for t, avg, mx in results:
        marker = "← best" if t == best_timeout else ""
        print(f"   {t:2d}   |    {avg:8.1f}   |    {mx:8.1f} {marker}")

if __name__ == "__main__":
    main()
