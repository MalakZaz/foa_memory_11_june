"""
===========================================================
 FILE: run_paper_ready.py
 MODULE: FULL PAPER-GRADE EXPERIMENT PIPELINE
===========================================================
"""

from src.utils.seed import set_seed

import json
import os
import numpy as np
import pandas as pd

from src.scenario.scenario_generator import generate_scenario

from src.baselines.random_search import run_random_search
from src.baselines.greedy import run_greedy
from src.baselines.pso_v0 import run_pso
from src.baselines.gwo_v0 import run_gwo

from src.foa.foa_v0 import run_foa
from src.core.fitness import fitness_function
from src.config import (
    N, M, RUNS, ITERATIONS, POP_SIZE,
    ETA, SEED
)

from experiments.stats.statistical_analysis import (
    statistical_report,
    print_report,
    plot_boxplot
)

from experiments.plots.convergence_plot import plot_convergence


def run_foa_batch(scenario, N, M, runs, iterations, pop_size, eta):

    results = []

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

    for _ in range(runs):

        _, fit, _ = run_foa(
            iterations=iterations,
            pop_size=pop_size,
            N=N,
            M=M,
            eta=eta,
            scenario=scenario,
            fitness_fn=fitness_fn
        )

        results.append(fit)

    return results


def run_baseline_batches(scenario, N, M, runs, iterations, pop_size, eta):

    foa_results = run_foa_batch(
        scenario,
        N,
        M,
        runs,
        iterations,
        pop_size,
        eta
    )

    random_results = []
    greedy_results = []
    pso_results = []
    gwo_results = []

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

    for _ in range(runs):

        _, rf = run_random_search(iterations, N, M, scenario)
        random_results.append(rf)

        _, gf = run_greedy(N, M, scenario)
        greedy_results.append(gf)

        _, pf, _ = run_pso(
            iterations=iterations,
            pop_size=pop_size,
            N=N,
            M=M,
            scenario=scenario,
            fitness_fn=fitness_fn
        )

        pso_results.append(pf)

        _, gwf, _ = run_gwo(
        iterations=iterations,
            pop_size=pop_size,
            N=N,
            M=M,
            scenario=scenario,
            fitness_fn=fitness_fn
        )

        gwo_results.append(gwf)


    return {
        "FOA": foa_results,
        "Random": random_results,
        "Greedy": greedy_results,
        "PSO": pso_results,
        "GWO": gwo_results              
    }

if __name__ == "__main__":

    set_seed(SEED)

    os.makedirs("results/v0", exist_ok=True)

    print("\n========================================")
    print(" FOA-CRN PAPER-READY EXPERIMENT ENGINE")
    print("========================================\n")

    

    scenario = generate_scenario(N, M, seed=SEED)

    print("\n========================================")
    print(" EXPERIMENT CONFIGURATION")
    print("========================================")
    print(f"N (SUs)           : {N}")
    print(f"M (Channels)      : {M}")
    print(f"P_TX              : {scenario['P_TX']}")
    print(f"SIGMA2            : {scenario['SIGMA2']}")
    print(f"SINR_MIN          : {scenario['SINR_MIN']}")

    print("\nQoS Parameters Usage:")
    print("  • P_TX and SIGMA2 are used in SINR computation")
    print("  • SINR_MIN is enforced through the fitness evaluation")
    print("  • R_MIN is enforced through the fitness evaluation")
    
    print(f"FOA Population    : {POP_SIZE}")
    print(f"FOA Iterations    : {ITERATIONS}")
    print(f"ETA               : {ETA}")
    print(f"Statistical Runs  : {RUNS}")
    print("----------------------------------------")
    print(f"Mean B            : {np.mean(scenario['B']):.2f}")
    print(f"Mean b            : {np.mean(scenario['b']):.2f}")
    print(f"Mean duration d   : {np.mean(scenario['d']):.2f}")
    print(f"Mean channel gain : {np.mean(scenario['h']):.4f}")
    print(f"B range           : [{np.min(scenario['B'])}, {np.max(scenario['B'])}]")
    print(f"b range           : [{np.min(scenario['b'])}, {np.max(scenario['b'])}]")
    print(f"d range           : [{np.min(scenario['d'])}, {np.max(scenario['d'])}]")
    print(f"h range           : [{np.min(scenario['h']):.3f}, {np.max(scenario['h']):.3f}]")
    print("========================================\n")

    results = run_baseline_batches(
    scenario,
    N,
    M,
    runs=RUNS,
    iterations=ITERATIONS,
    pop_size=POP_SIZE,
    eta=ETA
   )

    report = statistical_report(results)
    print_report(report)

    with open("results/v0/stats_report.json", "w") as f:
        json.dump(report, f, indent=4)

    plot_boxplot(results, "results/v0/boxplot_N{N}_M{M}_RUNS{RUNS}_ITERATIONS{ITERATIONS}.png")

    #plot_convergence()

    summary = {
        k: np.mean(v) for k, v in results.items()
    }

    pd.DataFrame([summary]).to_csv(
        "results/v0/summary_table.csv",
        index=False
    )

    print("\n========================================")
    print(" PIPELINE COMPLETED SUCCESSFULLY")
    print("========================================\n")