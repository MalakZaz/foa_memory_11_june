"""
foa_v0.py - Algorithme FOA standard (V0)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from config import POPULATION, ITERATIONS, RANDOM_SEED, N, M
from core.temporal import compute_overlap_matrix
from foa.population import initialize_population
from foa.movement import move_towards_best
from foa.evaluation import evaluate_population


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
        'time': elapsed_time
    }


def plot_convergence(history, save_path='convergence.png'):
    """Génère et sauvegarde la courbe de convergence."""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(history, 'b-', linewidth=2)
        plt.xlabel('Itérations')
        plt.ylabel('Fitness')
        plt.title('Convergence de FOA V0')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📊 Courbe sauvegardée: {save_path}")
    except ImportError:
        print("⚠️ Matplotlib non disponible")


def run_foa_v0(verbose=True):
    """Exécute FOA V0."""
    
    np.random.seed(RANDOM_SEED)
    O = compute_overlap_matrix()
    population = initialize_population()
    
    if verbose:
        print("=" * 60)
        print("FOA V0 - Allocation spectrale CRN")
        print("=" * 60)
        print(f"Population: {POPULATION}, Itérations: {ITERATIONS}")
    
    # Évaluation initiale
    fitnesses, allocations = evaluate_population(population, O, repair=True)
    
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
        # Déplacement
        new_population = []
        for pos in population:
            new_pos = move_towards_best(pos, best_position)
            new_population.append(new_pos)
        
        # Évaluation
        new_fitnesses, new_allocations = evaluate_population(new_population, O, repair=True)
        
        # Mise à jour du meilleur
        current_best_idx = np.argmax(new_fitnesses)
        if new_fitnesses[current_best_idx] > best_fitness:
            best_fitness = new_fitnesses[current_best_idx]
            best_position = new_population[current_best_idx].copy()
            best_allocation = new_allocations[current_best_idx].copy()
            if verbose:
                print(f"✨ Itération {iteration}: nouvelle meilleure fitness = {best_fitness:.4f}")
        elif verbose and iteration % 20 == 0:
            print(f"   Itération {iteration}: fitness = {best_fitness:.4f}")
        
        population = new_population
        fitnesses = new_fitnesses
        allocations = new_allocations
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
        
        # Courbe de convergence
        plot_convergence(history)
    
    stats = {
        'elapsed_time': elapsed_time,
        'metrics': metrics,
        'history': history
    }
    
    return best_fitness, best_allocation, history, stats


if __name__ == "__main__":
    run_foa_v0(verbose=True)