from src.utils.seed import set_seed
from src.scenario.scenario_generator import generate_scenario
from src.core.fitness import fitness_function

from src.foa.foa_v0 import run_foa
from src.foa.foa_memory_v2 import run_foa_memory_v2

from src.config import (
    N,
    M,
    ITERATIONS,
    POP_SIZE,
    ETA,
    SEED,
    BAD_THRESHOLD
)


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

    set_seed(SEED)

    _, fit_v0, _ = run_foa(
        iterations=ITERATIONS,
        pop_size=POP_SIZE,
        N=N,
        M=M,
        eta=ETA,
        scenario=scenario,
        fitness_fn=fitness_fn
    )

    set_seed(SEED)

    _, fit_v2, hist_v2, skipped, mem_size = run_foa_memory_v2(
        iterations=ITERATIONS,
        pop_size=POP_SIZE,
        N=N,
        M=M,
        eta=ETA,
        scenario=scenario,
        fitness_fn=fitness_fn,
        bad_threshold=BAD_THRESHOLD
    )

    print("\n========================================")
    print(" FOA-MEMORY V2 TEST")
    print("========================================")
    print(f"FOA V0 fitness        : {fit_v0:.4f}")
    print(f"FOA-Memory V2 fitness : {fit_v2:.4f}")
    print(f"Improvement (%)       : {((fit_v2 - fit_v0) / abs(fit_v0)) * 100:.2f}")
    print(f"Total skipped         : {sum(skipped)}")
    print(f"Avoidance rate (%)    : {(sum(skipped) / (ITERATIONS * POP_SIZE)) * 100:.2f}")
    print(f"Final memory size     : {mem_size[-1]}")
    print(f"History length        : {len(hist_v2)}")
    print("========================================")