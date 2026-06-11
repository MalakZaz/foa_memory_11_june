"""
===========================================================
 FILE: batch_memory_v2.py
 PURPOSE
-------
FOA V0 vs FOA-Memory V2 statistical comparison.
Also stores the best allocation and scenario configuration.
===========================================================
"""

import os
import json
import numpy as np

from src.utils.seed import set_seed
from src.scenario.scenario_generator import generate_scenario
from src.core.fitness import fitness_function

from src.foa.foa_v0 import run_foa
from src.foa.foa_memory_v2 import run_foa_memory_v2

from src.config import (
    N, M, RUNS, ITERATIONS, POP_SIZE,
    ETA, SEED, BAD_THRESHOLD
)


SAVE_DIR = "results/v0/best_runs"


def build_fitness(scenario):

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

    return fitness_fn


def save_best_run(prefix, best_info):

    os.makedirs(SAVE_DIR, exist_ok=True)

    np.save(
        os.path.join(SAVE_DIR, f"{prefix}_best_alpha.npy"),
        best_info["alpha"]
    )

    np.savez(
        os.path.join(SAVE_DIR, f"{prefix}_best_scenario.npz"),
        **best_info["scenario"]
    )

    metadata = {
        "run": best_info["run"],
        "seed": best_info["seed"],
        "fitness": float(best_info["fitness"]),
        "N": N,
        "M": M,
        "RUNS": RUNS,
        "ITERATIONS": ITERATIONS,
        "POP_SIZE": POP_SIZE,
        "ETA": ETA,
        "BAD_THRESHOLD": BAD_THRESHOLD
    }

    with open(
        os.path.join(SAVE_DIR, f"{prefix}_best_metadata.json"),
        "w"
    ) as f:
        json.dump(metadata, f, indent=4)


foa_results = []
memory_results = []

skipped_results = []
avoidance_results = []

best_foa_info = {
    "fitness": -np.inf
}

best_memory_info = {
    "fitness": -np.inf
}

print("\n========================================")
print(" FOA V0 vs FOA-Memory V2")
print("========================================\n")

for run in range(RUNS):

    seed = SEED + run

    scenario = generate_scenario(
        N,
        M,
        seed=seed
    )

    fitness_fn = build_fitness(scenario)

    set_seed(seed)

    best_alpha_foa, fit_foa, _ = run_foa(
        iterations=ITERATIONS,
        pop_size=POP_SIZE,
        N=N,
        M=M,
        eta=ETA,
        scenario=scenario,
        fitness_fn=fitness_fn
    )

    set_seed(seed)

    best_alpha_mem, fit_mem, _, skipped, _ = run_foa_memory_v2(
        iterations=ITERATIONS,
        pop_size=POP_SIZE,
        N=N,
        M=M,
        eta=ETA,
        scenario=scenario,
        fitness_fn=fitness_fn,
        bad_threshold=BAD_THRESHOLD
    )

    foa_results.append(fit_foa)
    memory_results.append(fit_mem)

    skipped_total = sum(skipped)

    skipped_results.append(skipped_total)

    avoidance_results.append(
        skipped_total / (ITERATIONS * POP_SIZE)
    )

    if fit_foa > best_foa_info["fitness"]:
        best_foa_info = {
            "fitness": fit_foa,
            "alpha": best_alpha_foa.copy(),
            "scenario": scenario.copy(),
            "run": run + 1,
            "seed": seed
        }

    if fit_mem > best_memory_info["fitness"]:
        best_memory_info = {
            "fitness": fit_mem,
            "alpha": best_alpha_mem.copy(),
            "scenario": scenario.copy(),
            "run": run + 1,
            "seed": seed
        }

    print(
        f"Run {run+1:02d} | "
        f"FOA={fit_foa:.1f} | "
        f"MEM={fit_mem:.1f} | "
        f"Skipped={skipped_total}"
    )


foa_results = np.array(foa_results)
memory_results = np.array(memory_results)

save_best_run("foa", best_foa_info)
save_best_run("memory", best_memory_info)

print("\n========================================")
print(" SUMMARY")
print("========================================")

print(
    f"FOA mean fitness        : "
    f"{foa_results.mean():.2f}"
)

print(
    f"Memory mean fitness     : "
    f"{memory_results.mean():.2f}"
)

print(
    f"Difference (%)          : "
    f"{100*(memory_results.mean()-foa_results.mean())/foa_results.mean():.2f}"
)

print(
    f"Average skipped evals   : "
    f"{np.mean(skipped_results):.2f}"
)

print(
    f"Average avoidance rate  : "
    f"{100*np.mean(avoidance_results):.2f}%"
)

print("----------------------------------------")
print(f"Best FOA run            : {best_foa_info['run']}")
print(f"Best FOA seed           : {best_foa_info['seed']}")
print(f"Best FOA fitness        : {best_foa_info['fitness']:.2f}")

print(f"Best Memory run         : {best_memory_info['run']}")
print(f"Best Memory seed        : {best_memory_info['seed']}")
print(f"Best Memory fitness     : {best_memory_info['fitness']:.2f}")

print("----------------------------------------")
print(f"Best run files saved to : {SAVE_DIR}")
print("========================================")