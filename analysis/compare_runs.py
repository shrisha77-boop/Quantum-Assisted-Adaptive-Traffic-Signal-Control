"""
Parses SUMO tripinfo.xml / summary.xml outputs and computes the metrics
needed to answer "does this method actually reduce congestion" --
average waiting time, travel time/timeLoss, throughput, and max queue.

Usage:
    Run each configuration (fixed-time, rule-based-only, classical-fallback,
    simulated-annealing, real-QPU), each with 3-5 random seeds, saving each
    run's outputs to a separate results folder. Then:

    python analysis/compare_runs.py \
        --baseline results_fixed_time/ \
        --method results_full_method_sa/ \
        --label "Simulated annealing tier"

Each results folder must contain tripinfo.xml (required) and optionally
summary.xml / queues.xml. To generate tripinfo.xml, add this to your
.sumocfg <output> section (or pass on the sumo/main_controller command line):

    <output>
        <tripinfo-output value="results/tripinfo.xml"/>
        <summary-output value="results/summary.xml"/>
        <queue-output value="results/queues.xml"/>
    </output>
"""

import argparse
import glob
import os
import statistics
import xml.etree.ElementTree as ET


def parse_tripinfo(path):
    """Returns list of per-vehicle dicts: duration, waitingTime, timeLoss."""
    tree = ET.parse(path)
    trips = []
    for trip in tree.getroot().findall("tripinfo"):
        trips.append({
            "duration": float(trip.get("duration")),
            "waitingTime": float(trip.get("waitingTime")),
            "timeLoss": float(trip.get("timeLoss")),
        })
    return trips


def parse_queue_max(path):
    """Returns the max queue length observed across all lanes/time in queues.xml."""
    if not os.path.exists(path):
        return None
    tree = ET.parse(path)
    max_q = 0
    for step in tree.getroot().findall("data"):
        for lane in step.findall(".//lane"):
            q = lane.get("queueing_length")
            if q is not None:
                max_q = max(max_q, float(q))
    return max_q


def summarize_run(results_dir):
    """Aggregates metrics for a single run (one seed) from its results folder."""
    tripinfo_path = os.path.join(results_dir, "tripinfo.xml")
    if not os.path.exists(tripinfo_path):
        raise FileNotFoundError(
            f"No tripinfo.xml in {results_dir} -- add <tripinfo-output> to your .sumocfg"
        )
    trips = parse_tripinfo(tripinfo_path)
    if not trips:
        raise ValueError(f"tripinfo.xml in {results_dir} contains no completed trips")

    queue_path = os.path.join(results_dir, "queues.xml")

    return {
        "num_completed_trips": len(trips),  # throughput proxy
        "avg_waiting_time": statistics.mean(t["waitingTime"] for t in trips),
        "avg_duration": statistics.mean(t["duration"] for t in trips),
        "avg_time_loss": statistics.mean(t["timeLoss"] for t in trips),
        "max_queue": parse_queue_max(queue_path),
    }


def summarize_multi_seed(results_dir_glob):
    """
    results_dir_glob: a glob pattern matching multiple seed-run folders,
    e.g. 'results_fixed_time_seed*/' -- aggregates mean +/- stdev across seeds.
    """
    dirs = sorted(glob.glob(results_dir_glob))
    if not dirs:
        raise FileNotFoundError(f"No directories matched: {results_dir_glob}")

    per_seed = [summarize_run(d) for d in dirs]
    metrics = per_seed[0].keys()
    aggregated = {}
    for m in metrics:
        values = [s[m] for s in per_seed if s[m] is not None]
        if not values:
            aggregated[m] = (None, None)
        elif len(values) == 1:
            aggregated[m] = (values[0], 0.0)
        else:
            aggregated[m] = (statistics.mean(values), statistics.stdev(values))
    return aggregated, len(dirs)


def pct_reduction(baseline, method):
    if baseline in (None, 0):
        return None
    return (baseline - method) / baseline * 100


def print_comparison(baseline_stats, method_stats, label, n_baseline, n_method):
    print(f"\n{'='*70}")
    print(f"Baseline (n={n_baseline} seeds)  vs  {label} (n={n_method} seeds)")
    print(f"{'='*70}")
    rows = [
        ("Avg waiting time (s)", "avg_waiting_time", True),
        ("Avg travel time (s)", "avg_duration", True),
        ("Avg time loss (s)", "avg_time_loss", True),
        ("Completed trips (throughput)", "num_completed_trips", False),
        ("Max queue length", "max_queue", True),
    ]
    for name, key, lower_is_better in rows:
        b_mean, b_std = baseline_stats[key]
        m_mean, m_std = method_stats[key]
        if b_mean is None or m_mean is None:
            print(f"{name:32s}  (no data)")
            continue
        change = pct_reduction(b_mean, m_mean) if lower_is_better else pct_reduction(m_mean, b_mean) * -1
        direction = "reduction" if lower_is_better else "increase"
        print(f"{name:32s}  baseline {b_mean:8.2f} ± {b_std:6.2f}   "
              f"method {m_mean:8.2f} ± {m_std:6.2f}   "
              f"({change:+.1f}% {direction})")
    print(f"{'='*70}")
    print("Note: with n<5 seeds per config, treat these as indicative, not")
    print("statistically confirmed -- run more seeds before citing exact %.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True,
                         help="Glob pattern for baseline results dirs, e.g. 'results_fixed_time_seed*/'")
    parser.add_argument("--method", required=True,
                         help="Glob pattern for method results dirs, e.g. 'results_full_method_seed*/'")
    parser.add_argument("--label", default="Method")
    args = parser.parse_args()

    baseline_stats, n_b = summarize_multi_seed(args.baseline)
    method_stats, n_m = summarize_multi_seed(args.method)
    print_comparison(baseline_stats, method_stats, args.label, n_b, n_m)
