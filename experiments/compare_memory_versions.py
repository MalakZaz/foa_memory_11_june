from src.config import (
    N,
    M,
    ITERATIONS,
    POP_SIZE,
    ETA,
    SEED,
    BAD_THRESHOLD
)

from src.utils.seed import set_seed
from src.scenario.scenario_generator import generate_scenario
from src.core.fitness import fitness_function

from src.foa.foa_v0 import run_foa
from src.foa.foa_memory_v2 import run_foa_memory_v2
from src.foa.foa_memory_v3 import run_foa_memory_v3


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

    # ==========================
    # FOA V0
    # ==========================
    set_seed(SEED)

    _, fit_v0, hist_v0 = run_foa(
        iterations=ITERATIONS,
        pop_size=POP_SIZE,
        N=N,
        M=M,
        eta=ETA,
        scenario=scenario,
        fitness_fn=fitness_fn
    )

    # ==========================
    # FOA-Memory V2
    # ==========================
    set_seed(SEED)

    _, fit_v2, hist_v2, skipped_v2, mem_v2 = run_foa_memory_v2(
        iterations=ITERATIONS,
        pop_size=POP_SIZE,
        N=N,
        M=M,
        eta=ETA,
        scenario=scenario,
        fitness_fn=fitness_fn,
        bad_threshold=BAD_THRESHOLD
    )

    # ==========================
    # FOA-Memory V3
    # ==========================
    set_seed(SEED)

    _, fit_v3, hist_v3, skipped_v3, occ_v3, mem_v3, nov_v3 = run_foa_memory_v3(
        iterations=ITERATIONS,
        pop_size=POP_SIZE,
        N=N,
        M=M,
        eta=ETA,
        scenario=scenario,
        fitness_fn=fitness_fn,
        bad_threshold=BAD_THRESHOLD
    )

    print("\n================================================")
    print(" MEMORY COMPARISON")
    print("================================================")

    print(f"{'Algorithm':<15} {'Fitness':<12} {'Skipped':<10}")

    print(f"{'FOA':<15} "
          f"{fit_v0:<12.2f} "
          f"{0:<10}")

    print(f"{'FOA-V2':<15} "
          f"{fit_v2:<12.2f} "
          f"{sum(skipped_v2):<10}")

    print(f"{'FOA-V3':<15} "
          f"{fit_v3:<12.2f} "
          f"{sum(skipped_v3):<10}")

    print("================================================")

    print(f"V2 Improvement (%): "
          f"{100*(fit_v2-fit_v0)/abs(fit_v0):.2f}")

    print(f"V3 Improvement (%): "
          f"{100*(fit_v3-fit_v0)/abs(fit_v0):.2f}")

    print(f"Final V3 Occupancy : {occ_v3[-1]:.4f}")

    print(f"Final V3 Novelty   : {nov_v3[-1]:.4f}")