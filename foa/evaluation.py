"""
evaluation.py - Évaluation d'une position FOA
"""

import numpy as np
from config import N, M, A
from core.temporal import compute_overlap_matrix
from core.constraints import check_constraints
from core.repair import repair_allocation
from core.fitness import compute_fitness


def evaluate_position(position, O=None, repair=True):
    """
    Pipeline complet d'évaluation.
    """
    # 1. Binarisation
    alpha = (position > 0.5).astype(int)
    
    # 2. Application disponibilité
    alpha = alpha * A
    
    # 3. Contrainte SU (1 canal max)
    for s in range(N):
        if np.sum(alpha[s, :]) > 1:
            # Garder le canal avec la plus grande valeur
            best_p = np.argmax(position[s])
            alpha[s, :] = 0
            if A[s, best_p] == 1:
                alpha[s, best_p] = 1
    
    # 4. Matrice de chevauchement
    if O is None:
        O = compute_overlap_matrix()
    
    # 5. Vérification des contraintes
    if not check_constraints(alpha, O):
        if repair:
            alpha = repair_allocation(alpha, O)
            if not check_constraints(alpha, O):
                return -np.inf, alpha
        else:
            return -np.inf, alpha
    
    # 6. Calcul de la fitness
    fitness = compute_fitness(alpha, O)
    
    return fitness, alpha


def evaluate_population(population, O=None, repair=True):
    """Évalue toute une population."""
    if O is None:
        O = compute_overlap_matrix()
    
    fitnesses = []
    allocations = []
    for pos in population:
        fit, alpha = evaluate_position(pos, O, repair)
        fitnesses.append(fit)
        allocations.append(alpha)
    
    return fitnesses, allocations