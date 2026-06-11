"""
===========================================================
 FILE: plot.py
 MODULE: FOA-CRN Scientific Visualization Engine
===========================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ===========================================================
# 1. CONVERGENCE CURVE
# ===========================================================
def plot_convergence(best_fitness_history):
    """
    Plot FOA convergence over iterations.
    """

    plt.figure()
    plt.plot(best_fitness_history)

    plt.title("FOA Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.grid(True)

    plt.show()


# ===========================================================
# 2. AVERAGE FITNESS EVOLUTION
# ===========================================================
def plot_average_fitness(avg_fitness_history):
    """
    Track population quality over time.
    """

    plt.figure()
    plt.plot(avg_fitness_history)

    plt.title("Average Fitness Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Average Fitness")
    plt.grid(True)

    plt.show()


# ===========================================================
# 3. FEASIBILITY RATE EVOLUTION
# ===========================================================
def plot_feasibility(feasibility_history):
    """
    Shows constraint satisfaction evolution.
    """

    plt.figure()
    plt.plot(feasibility_history)

    plt.title("Feasibility Rate Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Feasibility Rate")
    plt.ylim(0, 1)
    plt.grid(True)

    plt.show()


# ===========================================================
# 4. DIVERSITY EVOLUTION
# ===========================================================
def plot_diversity(diversity_history):
    """
    Measures exploration vs exploitation behavior.
    """

    plt.figure()
    plt.plot(diversity_history)

    plt.title("Population Diversity Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Diversity (Hamming Distance)")
    plt.grid(True)

    plt.show()