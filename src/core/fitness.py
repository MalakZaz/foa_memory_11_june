"""
===========================================================
 FILE: fitness.py
 MODULE: MKP Scientific Fitness Function (FOA-CRN V0)
===========================================================
"""

import numpy as np

from src.core.sinr import compute_sinr
from src.core.rate import compute_rate


# ===========================================================
# FITNESS FUNCTION (PURE MKP OBJECTIVE)
# ===========================================================
def fitness_function(alpha,
                     h,
                     B,
                     P_TX,
                     SIGMA2,
                     w,
                     d):
    """
    Pure utility-based MKP objective.

    NOTE:
    - Constraints handled in Layer 2 (NOT here)
    - This function assumes alpha is feasible
    """

    # STEP 1: Physical layer
    sinr = compute_sinr(alpha, h, P_TX, SIGMA2)
    rate = compute_rate(sinr, B)

    # STEP 2: MKP utility
    utility = np.sum(alpha * rate * w[:, None] * d[:, None])

    return utility