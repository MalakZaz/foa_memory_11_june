"""
===========================================================
 FILE: results_engine.py
 MODULE: FOA-CRN Results Analysis Engine
===========================================================

PURPOSE
-------
Transforms raw FOA outputs into scientific KPIs:

    - Best fitness
    - Average fitness
    - Feasibility rate
    - Convergence speed
    - Spectral efficiency gain
    - Constraint violation statistics

This module is designed for:
    - IEEE / Elsevier tables
    - Experimental section of papers
    - Reproducibility reports

===========================================================
"""

import numpy as np


# ===========================================================
# 1. BASIC KPI EXTRACTION
# ===========================================================
def compute_basic_kpis(best_history, avg_history):
    """
    Extract core optimization indicators.
    """

    kpis = {}

    kpis["best_final"] = float(best_history[-1])
    kpis["best_initial"] = float(best_history[0])

    kpis["avg_final"] = float(avg_history[-1])
    kpis["avg_mean"] = float(np.mean(avg_history))

    kpis["improvement"] = (
        kpis["best_final"] - kpis["best_initial"]
    )

    return kpis


# ===========================================================
# 2. CONVERGENCE SPEED INDEX
# ===========================================================
def convergence_speed(best_history):
    """
    Measures how fast FOA converges.

    Defined as iteration where 95% of final fitness is reached.
    """

    final = best_history[-1]
    threshold = 0.95 * final

    for i, v in enumerate(best_history):
        if v >= threshold:
            return i

    return len(best_history)


# ===========================================================
# 3. FEASIBILITY STATISTICS
# ===========================================================
def feasibility_stats(feasible_history):
    """
    Computes constraint satisfaction ratio.
    """

    feasible_history = np.array(feasible_history)

    return {
        "feasibility_rate": float(np.mean(feasible_history)),
        "violations_rate": float(1 - np.mean(feasible_history))
    }


# ===========================================================
# 4. SPECTRAL EFFICIENCY GAIN
# ===========================================================
def spectral_gain(rate_history):
    """
    Measures improvement in spectral efficiency.
    """

    return float(rate_history[-1] - rate_history[0])


# ===========================================================
# 5. GLOBAL REPORT GENERATOR
# ===========================================================
def generate_report(best_history,
                    avg_history,
                    feasible_history,
                    rate_history):
    """
    Full scientific summary of FOA experiment.
    """

    report = {}

    # KPI core
    report["kpis"] = compute_basic_kpis(best_history, avg_history)

    # convergence
    report["convergence_speed"] = convergence_speed(best_history)

    # feasibility
    report["feasibility"] = feasibility_stats(feasible_history)

    # spectral efficiency
    report["spectral_gain"] = spectral_gain(rate_history)

    return report


# ===========================================================
# 6. PRINT PAPER-READY TABLE
# ===========================================================
def print_report(report):
    """
    Clean output for article / thesis.
    """

    print("\n========== FOA-CRN RESULTS REPORT ==========\n")

    print("BEST FITNESS FINAL      :", report["kpis"]["best_final"])
    print("IMPROVEMENT             :", report["kpis"]["improvement"])
    print("AVERAGE FITNESS         :", report["kpis"]["avg_mean"])

    print("\n--- CONVERGENCE ---")
    print("CONVERGENCE ITERATION   :", report["convergence_speed"])

    print("\n--- FEASIBILITY ---")
    print("FEASIBILITY RATE        :", report["feasibility"]["feasibility_rate"])
    print("VIOLATION RATE          :", report["feasibility"]["violations_rate"])

    print("\n--- SPECTRAL EFFICIENCY ---")
    print("GAIN                    :", report["spectral_gain"])

    print("\n============================================\n")