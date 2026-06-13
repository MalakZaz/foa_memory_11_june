"""
===========================================================
 FILE: run_full_comparison_v3.py
 PURPOSE: Full comparison including FOA-Memory V3
===========================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import *

from src.utils.seed import set_seed
from src.scenario.scenario_generator import generate_scenario
from src.core.fitness import fitness_function

from src.foa.foa_v0 import run_foa
from src.foa.foa_memory_v2 import run_foa_memory_v2
from src.foa.foa_memory_v3 import run_foa_memory_v3

from src.baselines.pso_v0 import run_pso
from src.baselines.gwo_v0 import run_gwo
from src.baselines.random_search import run_random_search
from src.baselines.greedy import run_greedy


SAVE_DIR = "results/v3_full_comparison"

print(f"Creating output directory: {SAVE_DIR}")
os.makedirs(SAVE_DIR, exist_ok=True)

assert os.path.exists(SAVE_DIR), (
    f"Failed to create output directory: {SAVE_DIR}"
)


def save_scenario_config():

    config_dict = {
        "EXPERIMENT_NAME": EXPERIMENT_NAME,
        "SEED": SEED,
        "N": N,
        "M": M,
        "P_TX": P_TX,
        "SIGMA2": SIGMA2,
        "SINR_MIN": SINR_MIN,
        "R_MIN": R_MIN,
        "USE_QOS": USE_QOS,
        "DELTA_T": DELTA_T,
        "POP_SIZE": POP_SIZE,
        "ITERATIONS": ITERATIONS,
        "RUNS": RUNS,
        "ETA": ETA,
        "BAD_THRESHOLD": BAD_THRESHOLD,
        "BLOOM_SIZE": BLOOM_SIZE,
        "BLOOM_HASHES": BLOOM_HASHES,
        "BLOOM_DECAY": BLOOM_DECAY,
        "BLOOM_RECOVERY": BLOOM_RECOVERY,
        "BLOOM_FAMILIARITY_THRESHOLD": BLOOM_FAMILIARITY_THRESHOLD
    }

    path = os.path.join(
        SAVE_DIR,
        "scenario_config.json"
    )

    with open(path, "w") as f:
        json.dump(config_dict, f, indent=4)

    print(f"Scenario configuration saved to: {path}")


def print_experiment_header():

    print("\n" + "=" * 80)
    print(f" {EXPERIMENT_NAME}")
    print("=" * 80)

    parameters = [
        ("N", N, "Number of Secondary Users competing for spectrum resources."),
        ("M", M, "Number of available channels."),
        ("RUNS", RUNS, "Number of independent executions for statistical validation."),
        ("ITERATIONS", ITERATIONS, "Maximum number of optimization iterations."),
        ("POP_SIZE", POP_SIZE, "Number of candidate solutions per iteration."),
        ("ETA", ETA, "FOA exploration step controlling movement intensity."),
        ("SEED", SEED, "Base random seed for reproducibility."),
        ("P_TX", P_TX, "Transmission power used in SINR and rate computation."),
        ("SIGMA2", SIGMA2, "Noise power used in SINR computation."),
        ("SINR_MIN", SINR_MIN, "Minimum SINR required for QoS satisfaction."),
        ("R_MIN", R_MIN, "Minimum data rate required for QoS satisfaction."),
        ("USE_QOS", USE_QOS, "Enables or disables QoS-aware repair/evaluation."),
        ("BAD_THRESHOLD", BAD_THRESHOLD, "Threshold used to avoid familiar poor solutions."),
        ("BLOOM_SIZE", BLOOM_SIZE, "Number of memory traces in the Bloom filter."),
        ("BLOOM_HASHES", BLOOM_HASHES, "Number of hash projections per solution."),
        ("BLOOM_DECAY", BLOOM_DECAY, "Rate at which repeated solutions become familiar."),
        ("BLOOM_RECOVERY", BLOOM_RECOVERY, "Rate at which novelty recovers over time."),
        ("BLOOM_FAMILIARITY_THRESHOLD", BLOOM_FAMILIARITY_THRESHOLD, "Novelty threshold below which a solution is considered familiar."),
    ]

    print(f"{'Parameter':<30} {'Value':<15} {'Role'}")
    print("-" * 80)

    for name, value, role in parameters:
        print(f"{name:<30} {str(value):<15} {role}")

    print("-" * 80)
    print("Compared algorithms:")
    print("  - FOA")
    print("  - FOA-Memory V2")
    print("  - FOA-Memory V3")
    print("  - PSO")
    print("  - GWO")
    print("  - Random Search")
    print("  - Greedy")
    print("=" * 80)


def build_fitness_fn(scenario):

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


def random_history(iterations, N, M, scenario):

    history = []
    best_so_far = -np.inf

    for _ in range(iterations):

        _, fit = run_random_search(
            iterations=1,
            N=N,
            M=M,
            scenario=scenario
        )

        best_so_far = max(
            best_so_far,
            fit
        )

        history.append(best_so_far)

    return history


def greedy_history(iterations, N, M, scenario):

    _, fit = run_greedy(
        N,
        M,
        scenario
    )

    return [fit] * iterations


def plot_convergence(df):

    summary = (
        df.groupby(
            ["algorithm", "iteration"]
        )["best_fitness"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(11, 6),
        facecolor="white"
    )

    ax.set_facecolor("white")

    markers = {
        "FOA": "o",
        "FOA-Memory V2": "^",
        "FOA-Memory V3": "D",
        "PSO": "s",
        "GWO": "x",
        "Random Search": "*",
        "Greedy": "P"
    }

    for algo in summary["algorithm"].unique():

        data = summary[
            summary["algorithm"] == algo
        ]

        ax.plot(
            data["iteration"],
            data["best_fitness"],
            linewidth=2.2,
            marker=markers.get(algo, "o"),
            markevery=5,
            markersize=5,
            label=algo
        )

    ax.set_xlabel("Iteration", fontsize=12)

    ax.set_ylabel("Average Best Fitness", fontsize=12)

    ax.set_title(
        f"Average Convergence over {RUNS} Independent Runs",
        fontsize=14,
        fontweight="bold"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    ax.legend(
        frameon=True
    )

    fig.tight_layout()

    path = os.path.join(
        SAVE_DIR,
        f"convergence_all_N{N}_M{M}_RUNS{RUNS}_ITER{ITERATIONS}.png"
    )

    fig.savefig(
        path,
        dpi=300,
        facecolor="white",
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved convergence plot: {path}")


def plot_memory_metrics(memory_df):

    summary = (
        memory_df.groupby(
            ["algorithm", "iteration"]
        )["skipped"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
        facecolor="white"
    )

    ax.set_facecolor("white")

    for algo in summary["algorithm"].unique():

        data = summary[
            summary["algorithm"] == algo
        ]

        ax.plot(
            data["iteration"],
            data["skipped"],
            linewidth=2.2,
            marker="o",
            markevery=5,
            markersize=5,
            label=algo
        )

    ax.set_xlabel("Iteration", fontsize=12)

    ax.set_ylabel("Average Skipped Evaluations", fontsize=12)

    ax.set_title(
        "Memory-Based Evaluation Avoidance",
        fontsize=14,
        fontweight="bold"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    ax.legend(
        frameon=True
    )

    fig.tight_layout()

    path = os.path.join(
        SAVE_DIR,
        f"memory_skipped_N{N}_M{M}_RUNS{RUNS}_ITER{ITERATIONS}.png"
    )

    fig.savefig(
        path,
        dpi=300,
        facecolor="white",
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved memory plot: {path}")


def plot_v3_novelty(v3_df):

    summary = (
        v3_df.groupby("iteration")[
            ["bloom_occupancy", "mean_novelty"]
        ]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
        facecolor="white"
    )

    ax.set_facecolor("white")

    ax.plot(
        summary["iteration"],
        summary["bloom_occupancy"],
        linewidth=2.2,
        marker="o",
        markevery=5,
        markersize=5,
        label="Bloom occupancy"
    )

    ax.plot(
        summary["iteration"],
        summary["mean_novelty"],
        linewidth=2.2,
        marker="s",
        markevery=5,
        markersize=5,
        label="Mean novelty"
    )

    ax.set_xlabel("Iteration", fontsize=12)

    ax.set_ylabel("Value", fontsize=12)

    ax.set_title(
        "Time-Sensitive Bloom Memory Dynamics in FOA-Memory V3",
        fontsize=14,
        fontweight="bold"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    ax.legend(
        frameon=True
    )

    fig.tight_layout()

    path = os.path.join(
        SAVE_DIR,
        f"v3_bloom_dynamics_N{N}_M{M}_RUNS{RUNS}_ITER{ITERATIONS}.png"
    )

    fig.savefig(
        path,
        dpi=300,
        facecolor="white",
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved V3 novelty plot: {path}")


if __name__ == "__main__":

    print_experiment_header()

    save_scenario_config()

    rows = []
    memory_rows = []
    v3_rows = []
    final_results = []

    for run in range(RUNS):

        seed = SEED + run

        print(f"\nRun {run + 1}/{RUNS} | seed={seed}")

        scenario = generate_scenario(
            N,
            M,
            seed=seed
        )

        fitness_fn = build_fitness_fn(
            scenario
        )

        set_seed(seed)

        _, fit_foa, hist_foa = run_foa(
            ITERATIONS,
            POP_SIZE,
            N,
            M,
            ETA,
            scenario,
            fitness_fn
        )

        set_seed(seed)

        _, fit_v2, hist_v2, skipped_v2, mem_v2 = run_foa_memory_v2(
            ITERATIONS,
            POP_SIZE,
            N,
            M,
            ETA,
            scenario,
            fitness_fn,
            BAD_THRESHOLD
        )

        set_seed(seed)

        _, fit_v3, hist_v3, skipped_v3, occ_v3, mem_v3, nov_v3 = run_foa_memory_v3(
            ITERATIONS,
            POP_SIZE,
            N,
            M,
            ETA,
            scenario,
            fitness_fn,
            BAD_THRESHOLD
        )

        set_seed(seed)

        _, fit_pso, hist_pso = run_pso(
            ITERATIONS,
            POP_SIZE,
            N,
            M,
            scenario,
            fitness_fn
        )

        set_seed(seed)

        _, fit_gwo, hist_gwo = run_gwo(
            ITERATIONS,
            POP_SIZE,
            N,
            M,
            scenario,
            fitness_fn
        )

        set_seed(seed)

        hist_random = random_history(
            ITERATIONS,
            N,
            M,
            scenario
        )

        fit_random = hist_random[-1]

        hist_greedy = greedy_history(
            ITERATIONS,
            N,
            M,
            scenario
        )

        fit_greedy = hist_greedy[-1]

        histories = {
            "FOA": hist_foa,
            "FOA-Memory V2": hist_v2,
            "FOA-Memory V3": hist_v3,
            "PSO": hist_pso,
            "GWO": hist_gwo,
            "Random Search": hist_random,
            "Greedy": hist_greedy
        }

        for algo, history in histories.items():

            for iteration, value in enumerate(
                history,
                start=1
            ):

                rows.append({
                    "run": run + 1,
                    "seed": seed,
                    "iteration": iteration,
                    "algorithm": algo,
                    "best_fitness": value
                })

        for iteration in range(ITERATIONS):

            memory_rows.append({
                "run": run + 1,
                "seed": seed,
                "iteration": iteration + 1,
                "algorithm": "FOA-Memory V2",
                "skipped": skipped_v2[iteration],
                "memory_size": mem_v2[iteration]
            })

            memory_rows.append({
                "run": run + 1,
                "seed": seed,
                "iteration": iteration + 1,
                "algorithm": "FOA-Memory V3",
                "skipped": skipped_v3[iteration],
                "memory_size": mem_v3[iteration]
            })

            v3_rows.append({
                "run": run + 1,
                "seed": seed,
                "iteration": iteration + 1,
                "bloom_occupancy": occ_v3[iteration],
                "mean_novelty": nov_v3[iteration],
                "skipped": skipped_v3[iteration],
                "memory_size": mem_v3[iteration]
            })

        final_results.append({
            "run": run + 1,
            "seed": seed,
            "FOA": fit_foa,
            "FOA-Memory V2": fit_v2,
            "FOA-Memory V3": fit_v3,
            "PSO": fit_pso,
            "GWO": fit_gwo,
            "Random Search": fit_random,
            "Greedy": fit_greedy,
            "V2_skipped_total": sum(skipped_v2),
            "V3_skipped_total": sum(skipped_v3),
            "V3_final_occupancy": occ_v3[-1],
            "V3_final_novelty": nov_v3[-1]
        })

        print(
            f"  FOA={fit_foa:.2f} | "
            f"V2={fit_v2:.2f} | "
            f"V3={fit_v3:.2f} | "
            f"PSO={fit_pso:.2f} | "
            f"GWO={fit_gwo:.2f} | "
            f"Random={fit_random:.2f} | "
            f"Greedy={fit_greedy:.2f}"
        )

    df = pd.DataFrame(rows)
    memory_df = pd.DataFrame(memory_rows)
    v3_df = pd.DataFrame(v3_rows)
    final_df = pd.DataFrame(final_results)

    df.to_csv(
        os.path.join(
            SAVE_DIR,
            "convergence_histories.csv"
        ),
        index=False
    )

    memory_df.to_csv(
        os.path.join(
            SAVE_DIR,
            "memory_metrics.csv"
        ),
        index=False
    )

    v3_df.to_csv(
        os.path.join(
            SAVE_DIR,
            "v3_bloom_dynamics.csv"
        ),
        index=False
    )

    final_df.to_csv(
        os.path.join(
            SAVE_DIR,
            "final_results_per_run.csv"
        ),
        index=False
    )

    summary = (
        final_df
        .drop(
            columns=[
                "run",
                "seed"
            ]
        )
        .agg([
            "mean",
            "std",
            "min",
            "max"
        ])
        .T
    )

    summary.to_csv(
        os.path.join(
            SAVE_DIR,
            "summary_table.csv"
        )
    )

    with open(
        os.path.join(
            SAVE_DIR,
            "summary_table.json"
        ),
        "w"
    ) as f:

        json.dump(
            summary.to_dict(),
            f,
            indent=4
        )

    plot_convergence(df)

    plot_memory_metrics(memory_df)

    plot_v3_novelty(v3_df)

    print("\n================================================")
    print(" SUMMARY")
    print("================================================")
    print(summary)
    print("================================================")
    print(f"All results saved to: {SAVE_DIR}")
    print("================================================")