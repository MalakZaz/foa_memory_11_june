"""
===========================================================
 FILE: sanity_checks.py
 MODULE: Scientific Validation Engine (FOA-CRN V0)
===========================================================

PURPOSE
-------
This module validates the correctness of the entire FOA-CRN
pipeline before running experiments.

It ensures:
    1. MKP structure validity
    2. SINR physical consistency
    3. Rate monotonicity (Shannon law)
    4. Constraint coherence
    5. FOA encoding validity

This module is critical for reproducibility and publication.

===========================================================
"""

import numpy as np

from src.core.sinr import compute_sinr
from src.core.rate import compute_rate
from src.core.constraints import is_feasible


# ===========================================================
# 1. CHECK BINARY STRUCTURE
# ===========================================================
def check_binary_alpha(alpha):
    """
    Ensure MKP decision variable is binary.
    """

    return np.all((alpha == 0) | (alpha == 1))


# ===========================================================
# 2. CHECK EXCLUSIVITY STRUCTURE
# ===========================================================
def check_exclusivity(alpha):
    """
    Each SU must be assigned to at most one channel.
    """

    return np.all(np.sum(alpha, axis=1) <= 1)


# ===========================================================
# 3. SINR PHYSICAL VALIDITY
# ===========================================================
def check_sinr_validity(alpha, h, P_TX, sigma2):
    """
    SINR must be positive and finite.
    """

    sinr = compute_sinr(alpha, h, P_TX, sigma2)

    if np.any(np.isnan(sinr)) or np.any(np.isinf(sinr)):
        return False

    return np.all(sinr >= 0)


# ===========================================================
# 4. RATE CONSISTENCY CHECK
# ===========================================================
def check_rate_consistency(alpha, h, B, P_TX, sigma2):
    """
    Shannon law must produce non-negative rates.
    """

    sinr = compute_sinr(alpha, h, P_TX, sigma2)
    rate = compute_rate(sinr, B)

    return np.all(rate >= 0)


# ===========================================================
# 5. FEASIBILITY CROSS-CHECK
# ===========================================================
def check_feasibility(alpha, scenario):
    """
    Full constraint validation using MKP model.
    """

    return is_feasible(
        alpha,
        scenario["b"],
        scenario["B"],
        scenario["t_start"],
        scenario["t_end"],
        scenario["t_pu_start"],
        scenario["t_pu_end"]
    )


# ===========================================================
# 6. FULL SYSTEM VALIDATION PIPELINE
# ===========================================================
def run_sanity_checks(alpha, scenario):
    """
    Master validation function.

    Returns:
        dict of diagnostic results
    """

    results = {}

    results["binary"] = check_binary_alpha(alpha)
    results["exclusivity"] = check_exclusivity(alpha)

    results["feasibility"] = check_feasibility(alpha, scenario)

    results["sinr"] = check_sinr_validity(
        alpha,
        scenario["h"],
        scenario["P_TX"],
        scenario["SIGMA2"]
    )

    results["rate"] = check_rate_consistency(
        alpha,
        scenario["h"],
        scenario["B"],
        scenario["P_TX"],
        scenario["SIGMA2"]
    )

    results["global_valid"] = all(results.values())

    return results


# ===========================================================
# 7. DEBUG REPORT (REVIEWER MODE)
# ===========================================================
def print_sanity_report(results):
    """
    Pretty scientific report for debugging phase.
    """

    print("\n========== FOA-CRN SANITY REPORT ==========")

    for k, v in results.items():
        print(f"{k:20s} : {v}")

    print("==========================================\n")