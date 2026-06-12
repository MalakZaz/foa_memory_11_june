"""
===========================================================
 FILE: convergence_plot.py
 MODULE: Plot convergence curves including FOA-Memory V2
===========================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from src.config import N, M, RUNS, ITERATIONS

#CSV_PATH = f"results/v0/convergence_histories_memory_n{N}_M{M}_RUNS_{RUNS}_ITERATIONS_{ITERATIONS}.csv"
#SAVE_PATH = f"results/v0/convergence_plot_memory_n{N}_M{M}_RUNS_{RUNS}_ITERATIONS_{ITERATIONS}.png"

CSV_PATH = (
    f"results/v0/"
    f"convergence_histories_memory_"
    f"N{N}_M{M}_RUNS{RUNS}_ITERATIONS{ITERATIONS}.csv"
)

SAVE_PATH = (
    f"results/v0/"
    f"convergence_plot_memory_"
    f"N{N}_M{M}_RUNS{RUNS}_ITERATIONS{ITERATIONS}.png"
)

def load_data(csv_path):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    return pd.read_csv(csv_path)


def compute_summary(df):

    return (
        df.groupby(
            ["algorithm", "iteration"]
        )["best_fitness"]
        .agg(["mean", "std"])
        .reset_index()
    )


def plot_convergence():

    print("Loading convergence histories...")

    print("Expected CSV:", CSV_PATH)
    df = load_data(CSV_PATH)

    summary = compute_summary(df)

    fig, ax = plt.subplots(
        figsize=(11, 6),
        facecolor="white"
    )

    ax.set_facecolor("white")

    styles = {
        "FOA": {
            "marker": "o",
            "label": "FOA"
        },
        "FOA-Memory V2": {
            "marker": "^",
            "label": "FOA-Memory V2"
        },
        "PSO": {
            "marker": "s",
            "label": "PSO"
        },
        "Random Search": {
            "marker": "x",
            "label": "Random Search"
        },
        "Greedy": {
            "marker": "D",
            "label": "Greedy"
        }
    }

    final_results = []

    for algo in styles:

        data = summary[
            summary["algorithm"] == algo
        ]
        if data.empty:
            print(f"Warning: no data found for {algo}")
            continue

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

       # ax.fill_between(
        #x,
       # mean - std,
       # mean + std,
       # alpha=0.10
    #)
        final_results.append(
            f"{styles[algo]['label']}: "
            f"{mean.iloc[-1]:.1f}"
        )

    ax.set_xlabel(
        "Iteration",
        fontsize=12
    )

    ax.set_ylabel(
        "Average Best Fitness",
        fontsize=12
    )

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
        loc="lower right",
        frameon=True
    )

    summary_text = (
        "Final average fitness\n"
        + " | ".join(final_results)
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

    plt.tight_layout(
        rect=[0, 0.08, 1, 1]
    )

    print("Saving figure...")

    fig.savefig(
        SAVE_PATH,
        dpi=300,
        facecolor="white",
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\n========================================")
    print(" MEMORY CONVERGENCE PLOT SAVED")
    print("========================================")
    print(f"Saved to: {SAVE_PATH}")
    print("========================================")


if __name__ == "__main__":
    plot_convergence()