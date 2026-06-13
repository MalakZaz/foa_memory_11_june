"""
===========================================================
 FILE: run_v0.py
 MODULE: FOA-CRN Experimental Pipeline (CLEAN V0)
===========================================================
"""

# ===========================================================
# CORE
# ===========================================================
from src.core.fitness import fitness_function

# ===========================================================
# SCENARIO
# ===========================================================
from src.scenario.scenario_generator import generate_scenario

# ===========================================================
# FOA
# ===========================================================
from src.foa.foa_v0 import run_foa

# ===========================================================
# METRICS + LOGGING
# ===========================================================
from src.utils.logger import ExperimentLogger

from src.config import (
    N,
    M,
    ITERATIONS,
    POP_SIZE,
    ETA,
    SEED
)


# ===========================================================
# MAIN
# ===========================================================
if __name__ == "__main__":

    print("\n========================================")
    print(" FOA-CRN V0 EXPERIMENT (CLEAN RUN)")
    print("========================================\n")

    # -------------------------------------------------------
    # 1. SCENARIO GENERATION
    # -------------------------------------------------------
    scenario = generate_scenario(
        N,
        M,
        seed=SEED
    )

    # -------------------------------------------------------
    # 2. LOGGER
    # -------------------------------------------------------
    logger = ExperimentLogger()

    # -------------------------------------------------------
    # 3. FITNESS WRAPPER
    # -------------------------------------------------------
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

    # -------------------------------------------------------
    # 4. RUN FOA
    # -------------------------------------------------------
    best_alpha, best_fit, history = run_foa(
        iterations=ITERATIONS,
        pop_size=POP_SIZE,
        N=N,
        M=M,
        eta=ETA,
        scenario=scenario,
        fitness_fn=fitness_fn
    )

    # -------------------------------------------------------
    # 5. LOG CONVERGENCE
    # -------------------------------------------------------
    for iteration, fit in enumerate(history):

        avg_fit = fit * 0.8  # Placeholder

        logger.log_iteration(
            iteration,
            fit,
            avg_fit
        )

    # -------------------------------------------------------
    # 6. SAVE RESULTS
    # -------------------------------------------------------
    logger.set_final_best(best_fit)

    path = logger.save()

    # -------------------------------------------------------
    # 7. DISPLAY RESULTS
    # -------------------------------------------------------
    print("\n========================================")
    print(" FINAL RESULTS")
    print("========================================")

    print(f"Best fitness : {best_fit:.4f}")

    print(f"Iterations   : {ITERATIONS}")
    print(f"Population   : {POP_SIZE}")

    print(f"Results saved to:")
    print(path)

    print("========================================\n")