"""
===========================================================
 FILE: statistical_analysis.py
 MODULE: FOA-CRN Statistical Evaluation Engine
===========================================================

PURPOSE
-------
Provides rigorous statistical validation of FOA vs baselines:

    - Mean / Std / Variance
    - Confidence intervals
    - Hypothesis testing
    - Boxplot generation
    - Ranking of algorithms

Designed for IEEE / Elsevier experimental sections.
===========================================================
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# ===========================================================
# BASIC STATISTICS
# ===========================================================
def compute_stats(data):
    """
    Compute fundamental statistical indicators.
    """

    data = np.array(data)

    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "var": float(np.var(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data))
    }


# ===========================================================
# CONFIDENCE INTERVAL (95%)
# ===========================================================
def confidence_interval(data):
    """
    95% confidence interval for mean fitness.

    If the data have zero variance, the confidence interval
    collapses to the mean value.
    """

    data = np.array(data)

    mean = np.mean(data)

    if len(data) < 2 or np.std(data) == 0:
        return {
            "lower": float(mean),
            "upper": float(mean)
        }

    sem = stats.sem(data)

    ci = stats.t.interval(
        0.95,
        len(data) - 1,
        loc=mean,
        scale=sem
    )

    return {
        "lower": float(ci[0]),
        "upper": float(ci[1])
    }


# ===========================================================
# HYPOTHESIS TESTING (FOA vs baseline)
# ===========================================================
def t_test(foa, baseline):
    """
    Welch Student t-test for performance comparison.
    """

    foa = np.array(foa)
    baseline = np.array(baseline)

    if np.std(foa) == 0 and np.std(baseline) == 0:
        p_value = 1.0 if np.mean(foa) == np.mean(baseline) else 0.0
        return {
            "t_stat": 0.0,
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05)
        }

    t_stat, p_value = stats.ttest_ind(
        foa,
        baseline,
        equal_var=False
    )

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05)
    }


# ===========================================================
# NON-PARAMETRIC TEST (ROBUST CHECK)
# ===========================================================
def wilcoxon_test(foa, baseline):
    """
    Mann-Whitney U test.

    The alternative hypothesis is that FOA obtains larger
    fitness values than the compared baseline.
    """

    foa = np.array(foa)
    baseline = np.array(baseline)

    stat, p_value = stats.mannwhitneyu(
        foa,
        baseline,
        alternative="greater"
    )

    return {
        "u_stat": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05)
    }


# ===========================================================
# BOX PLOT GENERATOR
# ===========================================================
def plot_boxplot(results_dict, save_path="results/v0/boxplot.png"):
    """
    Generate a clean publication-ready comparison boxplot.
    """

    fig, ax = plt.subplots(
        figsize=(8, 5),
        facecolor="white"
    )

    ax.set_facecolor("white")

    labels = list(results_dict.keys())
    data = [results_dict[k] for k in labels]

    ax.boxplot(
        data,
        labels=labels
    )

    ax.set_title(
        "Statistical Comparison of FOA vs Baselines",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Fitness",
        fontsize=11
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        facecolor="white",
        bbox_inches="tight"
    )

    plt.close(fig)


# ===========================================================
# FULL REPORT GENERATOR
# ===========================================================
def statistical_report(results_dict):
    """
    Complete statistical evaluation.
    """

    report = {}

    report["basic_stats"] = {
        k: compute_stats(v)
        for k, v in results_dict.items()
    }

    report["confidence_intervals"] = {
        k: confidence_interval(v)
        for k, v in results_dict.items()
    }

    foa = results_dict["FOA"]

    report["tests"] = {
        k: t_test(foa, v)
        for k, v in results_dict.items()
        if k != "FOA"
    }

    report["wilcoxon"] = {
        k: wilcoxon_test(foa, v)
        for k, v in results_dict.items()
        if k != "FOA"
    }

    return report


# ===========================================================
# PRINT SCIENTIFIC SUMMARY
# ===========================================================
def print_report(report):

    print("\n========== STATISTICAL ANALYSIS ==========\n")

    print("--- BASIC STATS ---")
    for k, v in report["basic_stats"].items():
        print(k, v)

    print("\n--- SIGNIFICANCE TESTS ---")
    for k, v in report["tests"].items():
        print(k, v)

    print("\n--- WILCOXON TESTS ---")
    for k, v in report["wilcoxon"].items():
        print(k, v)

    print("\n==========================================\n")