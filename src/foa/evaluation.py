"""
===========================================================
 FILE: evaluation.py
 MODULE: FOA Candidate Evaluation Wrapper (V0)
===========================================================

DESCRIPTION
-----------
This module acts as a bridge between the FOA optimizer
and the MKP-based fitness function.

It ensures:
    - clean parameter passing
    - modular separation between FOA and CRN model
    - scalability for future FOA versions

===========================================================
"""

# ===========================================================
# IMPORTS
# ===========================================================
from core.fitness import fitness_function


# ===========================================================
# MAIN EVALUATION FUNCTION
# ===========================================================
def evaluate_candidate(alpha, scenario):
    """
    Evaluate a FOA candidate solution.

    Parameters
    ----------
    alpha : (N,M) binary allocation matrix
    scenario : dict containing all system parameters

    Returns
    -------
    fitness : float
        MKP-based utility value
    """

    return fitness_function(
        alpha=alpha,
        h=scenario["h"],
        b=scenario["b"],
        B=scenario["B"],
        t_start=scenario["t_start"],
        t_end=scenario["t_end"],
        t_pu_start=scenario["t_pu_start"],
        t_pu_end=scenario["t_pu_end"],
        P_TX=scenario["P_TX"],
        SIGMA2=scenario["SIGMA2"],
        SINR_MIN=scenario["SINR_MIN"],
        w=scenario["w"],
        d=scenario["d"]
    )