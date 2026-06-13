"""
===========================================================
 FILE: compare_foa_memory.py
 PURPOSE
-------
Compare the proposed FOA-Memory V2 against standard FOA.

Outputs:
    - results/v0/foa_vs_memory_convergence_*.csv
    - results/v0/foa_vs_memory_convergence_*.png
===========================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    N,
    M,
    RUNS,
    ITERATIONS,
    POP_SIZE,
    ETA,
    SEED,
    BAD_THRESHOLD,
    USE_QOS,
    SINR_MIN,
    R_MIN
)

from src.utils.seed import set_seed
from src.scenario.scenario_generator import generate_scenario
from src.core.fitness import fitness_function

from src.foa.foa_v0 import run_foa
from src.foa.foa_memory_v2 import run_foa_memory_v2


SAVE_DIR = "results/v0"

CSV_PATH = (
    f"{SAVE_DIR}/"
    f"foa_vs_memory_convergence_"
    f"N{N}_M{M}_RUNS{RUNS}_ITER{ITERATIONS}.csv"
)

FIG_PATH = (
    f"{SAVE_DIR}/"
    f"foa_vs_memory_convergence_"
    f"N{N}_M{M}_RUNS{RUNS}_ITER{ITERATIONS}.png"
)


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


def run_experiments():

    os.makedirs(SAVE_DIR, exist_ok=True)

    rows = []

    print("\n========================================")
    print(" FOA VS FOA-MEMORY V2 COMPARISON")
    print("========================================")
    print(f"N               : {N}")
    print(f"M               : {M}")
    print(f"Runs            : {RUNS}")
    print(f"Iterations      : {ITERATIONS}")
    print(f"Population      : {POP_SIZE}")
    print(f"ETA             : {ETA}")
    print(f"Bad threshold   : {BAD_THRESHOLD}")
    print(f"QoS enabled     : {USE_QOS}")

    if USE_QOS:
        print(f"SINR_MIN        : {SINR_MIN}")
        print(f"R_MIN           : {R_MIN}")

    print("========================================\n")

    skipped_totals = []

    for run in range(RUNS):

        seed = SEED + run

        print(f"Run {run + 1:02d}/{RUNS} | seed={seed}")

        scenario = generate_scenario(
            N,
            M,
            seed=seed
        )

        fitness_fn = build_fitness_fn(scenario)

        # -------------------------------
        # Standard FOA
        # -------------------------------
        set_seed(seed)

        _, foa_fit, foa_history = run_foa(
            iterations=ITERATIONS,
            pop_size=POP_SIZE,
            N=N,
            M=M,
            eta=ETA,
            scenario=scenario,
            fitness_fn=fitness_fn
        )

        # -------------------------------
        # Proposed FOA-Memory V2
        # -------------------------------
        set_seed(seed)

        _, memory_fit, memory_history, skipped, memory_size = run_foa_memory_v2(
            iterations=ITERATIONS,
            pop_size=POP_SIZE,
            N=N,
            M=M,
            eta=ETA,
            scenario=scenario,
            fitness_fn=fitness_fn,
            bad_threshold=BAD_THRESHOLD
        )

        skipped_total = sum(skipped)
        skipped_totals.append(skipped_total)

        print(
            f"   FOA={foa_fit:.2f} | "
            f"Memory={memory_fit:.2f} | "
            f"Skipped={skipped_total}"
        )

        for iteration in range(ITERATIONS):

            rows.append({
                "run": run + 1,
                "seed": seed,
                "iteration": iteration + 1,
                "algorithm": "FOA",
                "best_fitness": foa_history[iteration],
                "skipped": 0,
                "memory_size": 0
            })

            rows.append({
                "run": run + 1,
                "seed": seed,
                "iteration": iteration + 1,
                "algorithm": "FOA-Memory V2",
                "best_fitness": memory_history[iteration],
                "skipped": skipped[iteration],
                "memory_size": memory_size[iteration]
            })

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)

    average_skipped = sum(skipped_totals) / len(skipped_totals)
    avoidance_rate = average_skipped / (ITERATIONS * POP_SIZE) * 100

    print("\n========================================")
    print(" DATA SAVED")
    print("========================================")
    print(f"CSV saved to       : {CSV_PATH}")
    print(f"Rows               : {len(df)}")
    print(f"Average skipped    : {average_skipped:.2f}")
    print(f"Avoidance rate (%) : {avoidance_rate:.2f}")
    print("========================================\n")

    return df, average_skipped, avoidance_rate


def plot_comparison(df, average_skipped, avoidance_rate):

    summary = (
        df.groupby(["algorithm", "iteration"])["best_fitness"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
        facecolor="white"
    )

    ax.set_facecolor("white")

    styles = {
        "FOA": {
            "marker": "o",
            "label": "Standard FOA"
        },
        "FOA-Memory V2": {
            "marker": "^",
            "label": "Proposed FOA-Memory V2"
        }
    }

    final_results = []

    for algo in ["FOA", "FOA-Memory V2"]:

        data = summary[summary["algorithm"] == algo]

        x = data["iteration"]
        mean = data["mean"]
        std = data["std"].fillna(0)

        ax.plot(
            x,
            mean,
            linewidth=2.5,
            marker=styles[algo]["marker"],
            markevery=5,
            markersize=5,
            label=styles[algo]["label"]
        )

        #ax.fill_between(
          #  x,
           # mean - std,
          #  mean + std,
          #  alpha=0.10
        #)

        final_results.append(
            f"{styles[algo]['label']}: {mean.iloc[-1]:.2f}"
        )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Average Best Fitness", fontsize=12)

    ax.set_title(
        f"Standard FOA vs Proposed FOA-Memory over {RUNS} Runs",
        fontsize=14,
        fontweight="bold"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    ax.legend(
        loc="lower right",
        frameon=True
    )

    summary_text = (
        "Final average best fitness\n"
        + " | ".join(final_results)
        + f"\nAverage skipped evaluations: {average_skipped:.2f}"
        + f" | Avoidance rate: {avoidance_rate:.2f}%"
    )

    fig.text(
        0.5,
        0.02,
        summary_text,
        ha="center",
        fontsize=10,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round,pad=0.5"
        )
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    fig.savefig(
        FIG_PATH,
        dpi=300,
        facecolor="white",
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\n========================================")
    print(" FIGURE SAVED")
    print("========================================")
    print(f"Figure saved to: {FIG_PATH}")
    print("========================================\n")


if __name__ == "__main__":

    df, average_skipped, avoidance_rate = run_experiments()

    plot_comparison(df, average_skipped, avoidance_rate)