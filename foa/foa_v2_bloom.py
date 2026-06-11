"""
foa_v2_bloom.py - FOA avec mémoire Bloom (V2)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from config import POPULATION, ITERATIONS, N, M, RANDOM_SEED
from core.temporal import compute_overlap_matrix
from foa.population import initialize_population
from foa.movement import move_towards_best
from foa.evaluation import evaluate_population, evaluate_position
from memory.avoidance import AvoidanceManager


def compute_metrics(alpha, O, elapsed_time):
    """Calcule toutes les métriques d'évaluation."""
    from config import N, M, R_MIN, SINR_MIN, B_p
    from core.qos import check_qos
    
    total_sus = N
    served_sus = np.sum(np.sum(alpha, axis=1) > 0)
    served_percentage = (served_sus / total_sus) * 100
    total_allocated_channels = int(np.sum(alpha))
    channel_allocation_rate = (total_allocated_channels / (N * M)) * 100
    
    total_throughput = 0.0
    qos_satisfied = 0
    
    for s in range(N):
        user_qos = False
        for p in range(M):
            if alpha[s, p] == 1:
                ok, sinr, rate = check_qos(s, p, alpha, O)
                if ok:
                    total_throughput += rate
                    user_qos = True
        if user_qos:
            qos_satisfied += 1
    
    avg_throughput = total_throughput / served_sus if served_sus > 0 else 0.0
    qos_rate = (qos_satisfied / total_sus) * 100
    
    return {
        'N': total_sus,
        'served_sus': served_sus,
        'served_percentage': served_percentage,
        'allocated_channels': total_allocated_channels,
        'channel_rate': channel_allocation_rate,
        'total_throughput': total_throughput,
        'avg_throughput': avg_throughput,
        'qos_satisfied': qos_satisfied,
        'qos_rate': qos_rate,
        'execution_time': elapsed_time
    }


def run_foa_v2(verbose=True):
    """Exécute FOA avec mémoire Bloom (V2)."""
    
    np.random.seed(RANDOM_SEED)
    O = compute_overlap_matrix()
    population = initialize_population()
    avoidance = AvoidanceManager()
    
    if verbose:
        print("=" * 60)
        print("FOA V2 - Allocation spectrale CRN (avec mémoire Bloom)")
        print("=" * 60)
        print(f"Population: {POPULATION}, Itérations: {ITERATIONS}")
    
    # Évaluation initiale
    fitnesses, allocations = evaluate_population(population, O, repair=True)
    
    # Enregistrer les solutions initiales dans la mémoire
    for i, (pos, fit) in enumerate(zip(population, fitnesses)):
        if fit > -np.inf:
            indices = avoidance.bloom.hash_allocation(allocations[i])
            avoidance.record_evaluation(indices, fit, fit)
    
    best_idx = np.argmax(fitnesses)
    best_fitness = fitnesses[best_idx]
    best_position = population[best_idx].copy()
    best_allocation = allocations[best_idx].copy()
    
    history = [best_fitness]
    
    if verbose:
        print(f"Meilleure fitness initiale: {best_fitness:.4f}")
        print(f"Solutions valides: {sum(1 for f in fitnesses if f > -np.inf)}/{POPULATION}")
        print("\nDébut de l'optimisation...")
    
    start_time = time.time()
    
    for iteration in range(ITERATIONS):
        new_population = []
        new_fitnesses = []
        new_allocations = []
        
        for i in range(POPULATION):
            pos = population[i]
            alpha = allocations[i]
            fit = fitnesses[i]
            
            indices = avoidance.bloom.hash_allocation(alpha)
            
            if avoidance.should_avoid(fit, best_fitness, indices):
                # Évitement: nouvelle position aléatoire
                new_pos = np.random.rand(N, M)
                new_fit, new_alpha = evaluate_position(new_pos, O, repair=True)
            else:
                # Déplacement normal
                new_pos = move_towards_best(pos, best_position)
                new_fit, new_alpha = evaluate_position(new_pos, O, repair=True)
            
            new_indices = avoidance.bloom.hash_allocation(new_alpha)
            avoidance.record_evaluation(new_indices, new_fit, best_fitness)
            
            new_population.append(new_pos)
            new_fitnesses.append(new_fit)
            new_allocations.append(new_alpha)
        
        # Mise à jour
        population = new_population
        fitnesses = new_fitnesses
        allocations = new_allocations
        
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_position = population[current_best_idx].copy()
            best_allocation = allocations[current_best_idx].copy()
            if verbose:
                print(f"✨ Itération {iteration}: nouvelle meilleure fitness = {best_fitness:.4f}")
        elif verbose and iteration % 20 == 0:
            print(f"   Itération {iteration}: fitness = {best_fitness:.4f}")
        
        history.append(best_fitness)
    
    elapsed_time = time.time() - start_time
    
    # Métriques
    metrics = compute_metrics(best_allocation, O, elapsed_time)
    
    if verbose:
        print("\n" + "=" * 60)
        print("OPTIMISATION TERMINÉE")
        print("=" * 60)
        print(f"Meilleure fitness: {best_fitness:.4f}")
        print(f"Temps: {elapsed_time:.2f} s")
        print(f"Canaux alloués: {np.sum(best_allocation)}")
        
        print("\n📊 MÉTRIQUES D'ÉVALUATION")
        print("=" * 60)
        for k, v in metrics.items():
            if 'percentage' in k or 'rate' in k:
                print(f"{k}: {v:.1f}%")
            else:
                print(f"{k}: {v}")
        
        avoidance.print_stats()
    
    stats = {
        'elapsed_time': elapsed_time,
        'metrics': metrics,
        'avoidance_stats': avoidance.get_stats()
    }
    
    return best_fitness, best_allocation, history, stats


if __name__ == "__main__":
    run_foa_v2(verbose=True)