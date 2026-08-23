# Quantum-Assisted Adaptive Traffic Signal Control

Adaptive traffic signal management system for a Bengaluru road corridor implemented using SUMO, Classical Optimization, Simulated Annealing, and QAOA.

## Requirements

- Python 3.9 or above
- SUMO (Simulation of Urban MObility)
- Required Python libraries

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 1. Run the Default SUMO Baseline

Run the traffic simulation using SUMO's default fixed-time signal controller:

```bash
sumo -c scenario/config/corridor.sumocfg
```

To view the default simulation in SUMO-GUI:

```bash
sumo-gui -c scenario/config/corridor.sumocfg
```

---

## 2. Run the Proposed Adaptive Controller

Run the proposed adaptive system (which dynamically selects solvers and actuates green/yellow phase transitions):

With GUI:

```bash
python main_controller.py --gui --solver adaptive
```

Without GUI:

```bash
python main_controller.py --solver adaptive
```

You can also specify particular solvers:
- `--solver simulated_annealing` (Heuristic optimization)
- `--solver qaoa` (Quantum approximate optimization algorithm)
- `--solver adaptive` (Dynamic traffic-load solver policy: SA for light traffic, QAOA for heavy traffic)

---

## 3. Compare Results & Plot Performance

After running the baseline and proposed controller:

```bash
python analysis/plot_comparison.py
```

Or run multi-seed statistical comparisons:

```bash
python analysis/compare_runs.py --baseline "simulation/results_baseline/" --method "simulation/results/" --label "Adaptive QUBO Controller"
```

---

## Notes

- Simulation decision logs are output to `simulation/results/decision_results.jsonl`.
- Verification of QUBO formulation can be run anytime with `python tools/verify_qubo_and_solvers.py`.
