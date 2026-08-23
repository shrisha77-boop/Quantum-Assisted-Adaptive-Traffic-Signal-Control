"""
verify_qubo_and_solvers.py
===========================

Verification script to test QUBO Builder and Solvers:
Confirms that candidate phases with HIGH demand receive lower (more negative) energy
and are correctly selected by Classical and Simulated Annealing solvers.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from modules.qubo_builder import QUBOBuilder


from modules.qubo_solver.simulated_annealing_solver import solve as sa_solve
from modules.qubo_solver.solver_manager import SolverManager, SolverManagerConfig

def test_qubo_objective_and_solvers():
    builder = QUBOBuilder()

    # Test Scenario: NS has HIGH demand (waiting_time=50.0), EW has ZERO demand (0.0)
    decision_input = {
        "junction_id": "TestJunction",
        "candidate_roads": ["NS", "EW"],
        "waiting_time": {"NS": 50.0, "EW": 0.0},
        "queue_storage_ratio": {"NS": 0.5, "EW": 0.0},
        "queue_length": {"NS": 10.0, "EW": 0.0},
    }

    qubo = builder.build(decision_input)
    print("QUBO Matrix:", qubo["qubo_matrix"])
    print("Linear terms:", qubo["linear_terms"])


    # Test Simulated Annealing Solver
    res_sa = sa_solve(qubo)
    print("SA Result:", res_sa["selected_phase"], "Energy:", res_sa["best_energy"])

    # Assert NS (busy phase) is selected by both
    assert res_classical["selected_phase"] == "NS", f"Classical failed: selected {res_classical['selected_phase']}"
    assert res_sa["selected_phase"] == "NS", f"SA failed: selected {res_sa['selected_phase']}"

    # Test Adaptive Solver Manager
    mgr = SolverManager(SolverManagerConfig(solver="adaptive"))
    res_adaptive = mgr.solve(qubo, vehicle_count=10)
    print("Adaptive (N=10) Result:", res_adaptive["selected_phase"], "Solver:", res_adaptive["solver"])
    assert res_adaptive["selected_phase"] == "NS"

    print("\nSUCCESS: All QUBO objective and solver tests passed! Busy approach 'NS' correctly selected!")

if __name__ == "__main__":
    test_qubo_objective_and_solvers()
