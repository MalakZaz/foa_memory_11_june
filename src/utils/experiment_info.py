"""
===========================================================
 FILE: experiment_info.py
 PURPOSE: Experiment metadata, scenario explanation,
          problem formulation, and figure annotations
===========================================================
"""

from datetime import datetime
import platform

from src.config import (
    N,
    M,
    RUNS,
    ITERATIONS,
    POP_SIZE,
    ETA,
    USE_QOS,
    SINR_MIN,
    R_MIN,
    BAD_THRESHOLD,
    P_TX,
    SIGMA2
)


def get_execution_info():
    now = datetime.now()

    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "platform": platform.system(),
        "python": platform.python_version()
    }


def print_execution_info():
    info = get_execution_info()

    print("\n" + "=" * 80)
    print("EXPERIMENT INFORMATION")
    print("=" * 80)

    for key, value in info.items():
        print(f"{key.capitalize():<15}: {value}")

    print("=" * 80)


def print_scenario():
    print("\n" + "=" * 80)
    print("SCENARIO CONFIGURATION")
    print("=" * 80)

    rows = [
        ("N", N, "Number of Secondary Users competing for spectrum."),
        ("M", M, "Number of available channels."),
        ("P_TX", P_TX, "Transmission power used by Secondary Users."),
        ("SIGMA2", SIGMA2, "Background noise power."),
        ("USE_QOS", USE_QOS, "Enables SINR and rate constraints."),
        ("SINR_MIN", SINR_MIN, "Minimum acceptable signal quality."),
        ("R_MIN", R_MIN, "Minimum acceptable data rate."),
        ("POP_SIZE", POP_SIZE, "Number of candidate solutions per iteration."),
        ("ITERATIONS", ITERATIONS, "Maximum number of optimization iterations."),
        ("RUNS", RUNS, "Number of independent runs for statistical validation."),
        ("ETA", ETA, "FOA movement step size."),
        ("BAD_THRESHOLD", BAD_THRESHOLD, "Memory threshold for avoiding poor familiar solutions.")
    ]

    print(f"{'Parameter':<18}{'Value':<15}Description")
    print("-" * 80)

    for name, value, desc in rows:
        print(f"{name:<18}{str(value):<15}{desc}")

    print("=" * 80)


def print_experiment_description():
    print("\n" + "=" * 80)
    print("EXPERIMENT DESCRIPTION")
    print("=" * 80)

    print(
        "\nThis experiment studies dynamic spectrum allocation in "
        "Overlay Cognitive Radio Networks."
    )

    print(
        "\nThe goal is to assign available wireless channels to "
        "Secondary Users while maximizing network utility and "
        "respecting minimum Quality of Service requirements."
    )

    print(
        "\nThe proposed method is FOA-Memory V2, a memory-enhanced "
        "Fruit Fly Optimization Algorithm. The memory mechanism avoids "
        "re-evaluating familiar poor solutions, thereby reducing "
        "computational effort while preserving optimization quality."
    )

    print("\nIn simple terms:")
    print("  • Users ask for wireless resources.")
    print("  • Channels are limited.")
    print("  • The algorithm decides which user gets which channel.")
    print("  • The goal is to obtain high utility while respecting QoS.")
    print("  • The memory module avoids wasting time on bad repeated solutions.")

    print("=" * 80)


def print_problem_formulation():
    print("\n" + "=" * 80)
    print("OPTIMIZATION PROBLEM FORMULATION")
    print("=" * 80)

    print("\nOBJECTIVE FUNCTION")
    print("-" * 80)

    print("\nMaximize:")

    print("\n    U = sum_s sum_p x[s,p] * R[s,p] * w[s] * d[s]")

    print("\nwhere:")
    print("  x[s,p] : 1 if channel p is assigned to user s, otherwise 0.")
    print("  R[s,p] : achievable data rate of user s on channel p.")
    print("  w[s]   : priority weight of user s.")
    print("  d[s]   : requested service duration of user s.")

    print("\nThe objective favors:")
    print("  • high-rate allocations;")
    print("  • high-priority users;")
    print("  • longer requested service durations.")

    print("\nPHYSICAL LAYER MODEL")
    print("-" * 80)

    print("\nSINR model:")

    print(
        "\n    SINR[s,p] = (P_TX * h[s,p]) / "
        "(SIGMA2 + interference)"
    )

    print("\nAchievable rate model:")

    print("\n    R[s,p] = B[p] * log2(1 + SINR[s,p])")

    print(
        "\nBecause the current implementation enforces one SU per channel, "
        "SU-SU interference is avoided in feasible allocations."
    )

    print("\nCONSTRAINTS ENFORCED IN THE IMPLEMENTATION")
    print("-" * 80)

    constraints = [
        (
            "Channel exclusivity",
            "sum_s x[s,p] <= 1",
            "Each channel can be assigned to at most one SU."
        ),
        (
            "Bandwidth capacity",
            "sum_s x[s,p] * b[s] <= B[p]",
            "The assigned bandwidth demand cannot exceed channel capacity."
        ),
        (
            "SINR QoS",
            "SINR[s,p] >= SINR_MIN",
            "Allocated users must satisfy a minimum signal quality."
        ),
        (
            "Rate QoS",
            "R[s,p] >= R_MIN",
            "Allocated users must satisfy a minimum data rate."
        ),
        (
            "Binary allocation",
            "x[s,p] in {0,1}",
            "Allocation decisions are binary."
        )
    ]

    for title, formula, explanation in constraints:
        print(f"\n{title}")
        print(f"  Formula     : {formula}")
        print(f"  Explanation : {explanation}")

    print("\nMEMORY MECHANISM")
    print("-" * 80)

    print("\nA familiar solution is skipped when:")

    print("\n    f_stored < BAD_THRESHOLD * f_best")

    print("\nwhere:")
    print("  f_stored : fitness previously observed for the solution.")
    print("  f_best   : best fitness found so far.")
    print("  BAD_THRESHOLD : memory aggressiveness parameter.")

    print(
        "\nThis allows the optimizer to avoid repeated poor solutions "
        "without changing the original optimization problem."
    )

    print("=" * 80)


def print_metrics():
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)

    metrics = [
        ("Final fitness", "Best fitness obtained at the end of a run."),
        ("Mean fitness", "Average final fitness over all independent runs."),
        ("Standard deviation", "Variability of final fitness across runs."),
        ("Minimum fitness", "Worst final result observed."),
        ("Maximum fitness", "Best final result observed."),
        ("Confidence interval", "Uncertainty range of the mean fitness."),
        ("Student t-test", "Parametric statistical comparison against FOA."),
        ("Mann-Whitney U test", "Non-parametric robustness comparison."),
        ("Convergence curve", "Evolution of best fitness over iterations."),
        ("Boxplot", "Distribution of final fitness values."),
        ("Skipped evaluations", "Number of candidate solutions avoided by memory."),
        ("Avoidance rate", "Percentage of avoided fitness evaluations.")
    ]

    for i, (name, desc) in enumerate(metrics, 1):
        print(f"{i:2d}. {name:<25} {desc}")

    print("=" * 80)


def get_figure_footer(include_time=False):
    info = get_execution_info()

    footer = (
        f"N={N}, M={M} | Runs={RUNS} | Iter={ITERATIONS} | "
        f"QoS={USE_QOS}"
    )

    if include_time:
        footer += f" | Generated: {info['date']} {info['time']}"

    return footer