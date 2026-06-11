# ============================================================
# FILE: main.py
# PROJECT: FOA V0 - Cognitive Radio Network Spectrum Allocation
# AUTHOR: Research Implementation (Q1-grade)
# DESCRIPTION:
#   Main execution pipeline of FOA V0 for centralized CRN
#   spectrum scheduling with SINR-aware fitness evaluation.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================
import numpy as np

from src.config import *
from config.src.core.temporal import compute_overlap
from config.src.core.sinr import compute_sinr
from config.src.core.rate import compute_rate
from config.src.core.fitness import fitness

from src.foa.foa_v0 import init, binarize, update
from src.utils.metrics import compute_metrics
from src.utils.plot import plot_fitness


# ============================================================
# PRINT ALL SYSTEM PARAMETERS (IMPORTANT FOR REPRODUCIBILITY)
# ============================================================
print("\n================ FOA V0 PARAMETERS ================\n")

print(f"Number of SUs (N)          : {N}")
print(f"Number of Channels (M)     : {M}")
print(f"Population size            : {POP_SIZE}")
print(f"Iterations                 : {ITERATIONS}")
print(f"Transmission power (P_TX)  : {P_TX}")
print(f"Noise power (SIGMA^2)      : {SIGMA2}")
print(f"SINR threshold             : {SINR_MIN}")
print(f"Rate threshold             : {R_MIN}")
print(f"Time window (DELTA_T)      : {DELTA_T}")
print(f"FOA exploration factor η    : {ETA}")

print("\n===================================================\n")


# ============================================================
# MAIN FUNCTION
# ============================================================
def main():

    # --------------------------------------------------------
    # STEP 1: Generate CRN scenario (channels + users)
    # --------------------------------------------------------
    h, A, d, t, w = generate_scenario(N, M)

    # --------------------------------------------------------
    # STEP 2: Compute temporal overlap matrix
    # --------------------------------------------------------
    O = compute_overlap(t, d)

    # --------------------------------------------------------
    # STEP 3: Initialize FOA population
    # --------------------------------------------------------
    X = init(POP_SIZE, N, M)

    best_fit = -np.inf
    best_X = None
    history = []

    # --------------------------------------------------------
    # STEP 4: FOA optimization loop
    # --------------------------------------------------------
    for it in range(ITERATIONS):

        # Evaluate all solutions in population
        for i in range(POP_SIZE):

            # Convert continuous FOA position → binary allocation
            alpha = binarize(X[i])

            # Apply availability constraint (PU protection)
            alpha = alpha * A

            # Compute SINR based on interference + overlap
            sinr = compute_sinr(alpha, h, O)

            # Compute achievable data rate (Shannon)
            rate = compute_rate(sinr, alpha)

            # Compute fitness (network utility)
            fit = fitness(alpha, sinr, rate, w, d, DELTA_T, SINR_MIN)

            # Update global best solution
            if fit > best_fit:
                best_fit = fit
                best_X = X[i].copy()

        # FOA position update (exploration + exploitation)
        X = update(X, best_X, ETA)

        # Store convergence history
        history.append(best_fit)

        # Iteration logging (for debugging + paper traceability)
        print(f"[Iteration {it}] Best Fitness = {best_fit:.6f}")

    # --------------------------------------------------------
    # STEP 5: Extract best solution
    # --------------------------------------------------------
    alpha_best = binarize(best_X) * A
    sinr_best = compute_sinr(alpha_best, h, O)

    # --------------------------------------------------------
    # STEP 6: Compute final metrics
    # --------------------------------------------------------
    metrics = compute_metrics(alpha_best, sinr_best)

    print("\n================ FINAL METRICS ================\n")
    for k, v in metrics.items():
        print(f"{k} : {v}")

    # --------------------------------------------------------
    # STEP 7: Plot convergence curve
    # --------------------------------------------------------
    plot_fitness(history)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()