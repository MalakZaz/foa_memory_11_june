"""
===========================================================
 FILE: test_foa_memory_v4_niching.py
 PURPOSE: Quick test for FOA-Memory V4 Niching
===========================================================
"""

from src.config import (
    N,
    M,
    ITERATIONS,
    POP_SIZE,
    ETA,
    SEED,
    BAD_THRESHOLD,
    NUM_NICHES
)

from src.utils.seed import set_seed
from src.scenario.scenario_generator import generate_scenario
from src.core.fitness import fitness_function
from src.foa.foa_memory_v4_niching import run_foa_memory_v4_niching


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

    (
        best_alpha,
        best_fit,
        history,
        skipped,
        bloom_occupancy,
        memory_size,
        novelty_history
    ) = run_foa_memory_v4_niching(
        iterations=ITERATIONS,
        pop_size=POP_SIZE,
        N=N,
        M=M,
        eta=ETA,
        scenario=scenario,
        fitness_fn=fitness_fn,
        num_niches=NUM_NICHES,
        bad_threshold=BAD_THRESHOLD
    )

    print("\n========================================")
    print(" FOA-MEMORY V4 NICHING TEST")
    print("========================================")
    print(f"N                     : {N}")
    print(f"M                     : {M}")
    print(f"Population            : {POP_SIZE}")
    print(f"Iterations            : {ITERATIONS}")
    print(f"Niches                : {NUM_NICHES}")
    print(f"BAD_THRESHOLD         : {BAD_THRESHOLD}")
    print("----------------------------------------")
    print(f"Best fitness          : {best_fit:.2f}")
    print(f"History length        : {len(history)}")
    print(f"Total skipped         : {sum(skipped)}")
    print(f"Avoidance rate (%)    : {100 * sum(skipped) / (ITERATIONS * POP_SIZE):.2f}")
    print(f"Final Bloom occupancy : {bloom_occupancy[-1]:.4f}")
    print(f"Final memory size     : {memory_size[-1]}")
    print(f"Final mean novelty    : {novelty_history[-1]:.4f}")
    print("========================================")