from src.utils.seed import set_seed
from src.scenario.scenario_generator import generate_scenario
from src.core.fitness import fitness_function
from src.foa.foa_memory_v1 import run_foa_memory_v1


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


best_alpha, best_fit, history, skipped, mem_size = run_foa_memory_v1(
    iterations=50,
    pop_size=30,
    N=N,
    M=M,
    eta=0.1,
    scenario=scenario,
    fitness_fn=fitness_fn
)

print("Best fitness:", best_fit)
print("Final memory size:", mem_size[-1])
print("Total skipped:", sum(skipped))
print("History length:", len(history))