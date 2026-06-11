from src.utils.seed import set_seed
from src.scenario.scenario_generator import generate_scenario
from src.core.fitness import fitness_function

from src.foa.foa_v0 import run_foa
from src.foa.foa_memory_v2 import run_foa_memory_v2


set_seed(42)

N, M = 20, 10
scenario = generate_scenario(N, M, seed=42)


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

set_seed(42)

_, fit_v0, _ = run_foa(
    iterations=50,
    pop_size=30,
    N=N,
    M=M,
    eta=0.1,
    scenario=scenario,
    fitness_fn=fitness_fn
)

set_seed(42)

_, fit_v2, hist_v2, skipped, mem_size = run_foa_memory_v2(
    iterations=50,
    pop_size=30,
    N=N,
    M=M,
    eta=0.1,
    scenario=scenario,
    fitness_fn=fitness_fn,
    bad_threshold=0.8
)

print("FOA V0 fitness      :", fit_v0)
print("FOA-Memory V2 fitness:", fit_v2)
print("Improvement (%)     :", ((fit_v2 - fit_v0) / abs(fit_v0)) * 100)
print("Total skipped       :", sum(skipped))
print("Avoidance rate (%)  :", (sum(skipped) / (50 * 30)) * 100)
print("Final memory size   :", mem_size[-1])
print("History length      :", len(hist_v2))