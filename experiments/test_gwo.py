"""
===========================================================
 FILE: test_gwo.py
 PURPOSE: Quick test for GWO baseline
===========================================================
"""

from src.config import (
    N,
    M,
    ITERATIONS,
    POP_SIZE,
    SEED,
    USE_QOS,
    SINR_MIN,
    R_MIN
)

from src.utils.seed import set_seed
from src.scenario.scenario_generator import generate_scenario
from src.core.fitness import fitness_function
from src.baselines.gwo_v0 import run_gwo


if __name__ == "__main__":

    set_seed(SEED)

    scenario = generate_scenario(
        N,
        M,
        seed=SEED
    )

    def fitness_fn(alpha, **_):
        return fitness_function(
            alpha,
            h=scenario["h"],
            B=scenario["B"],
            P_TX=scenario["P_TX"],
            SIGMA2=scenario["SIGMA2"],
            w=scenario["w"],
            d=scenario["d"]
        )

    _, best_fit, history = run_gwo(
        iterations=ITERATIONS,
        pop_size=POP_SIZE,
        N=N,
        M=M,
        scenario=scenario,
        fitness_fn=fitness_fn
    )

    print("\n========================================")
    print(" GWO TEST")
    print("========================================")
    print(f"N               : {N}")
    print(f"M               : {M}")
    print(f"Population      : {POP_SIZE}")
    print(f"Iterations      : {ITERATIONS}")
    print(f"QoS Enabled     : {USE_QOS}")

    if USE_QOS:
        print(f"SINR_MIN        : {SINR_MIN}")
        print(f"R_MIN           : {R_MIN}")

    print("----------------------------------------")
    print(f"GWO best fitness: {best_fit:.2f}")
    print(f"History length  : {len(history)}")
    print("========================================")