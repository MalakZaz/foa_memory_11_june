"""
foa_v3_complete.py - FOA complet avec niching + mémoire Bloom (V3)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from config import POPULATION, ITERATIONS, N, M, RANDOM_SEED, SIGMA_SHARE, ALPHA_SHARE
from core.temporal import compute_overlap_matrix
from foa.population import initialize_population
from foa.movement import move_towards_best, compute_distance
from foa.evaluation import evaluate_population, evaluate_position
from memory.avoidance import AvoidanceManager


def sharing_function(d, sigma_share=SIGMA_SHARE, alpha=ALPHA_SHARE):
    """Fonction de partage pour le niching."""
    if d >= sigma_share:
        return 0.0
    return 1 - (d / sigma_share) ** alpha


def compute_shared_fitness(population, raw_fitnesses, sigma_share=SIGMA_SHARE, alpha=ALPHA_SHARE):
    """Calcule la fitness partagée pour toute la population."""
    n = len(population)
    shared = []
    for i in range(n):
        if raw_fitnesses[i] == -np.inf:
            shared.append(-np.inf)
            continue
        sum_sharing = 0.0
        for j in range(n):
            if i == j:
                continue
            dist = compute_distance(population[i], population[j])
            sum_sharing += sharing_function(dist, sigma_share, alpha)
        if sum_sharing > 0:
            shared.append(raw_fitnesses[i] / sum_sharing)
        else:
            shared.append(raw_fitnesses[i])
    return shared


def compute_metrics(alpha, O, elapsed_time):
    """Calcule toutes les métriques d'évaluation."""
    from config import N, M
    from core.qos import check_qos
    
    total_sus = N
    served_sus = int(np.sum(np.sum(alpha, axis=1) > 0))
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
    
    # Évitement : taux de collisions évitées
    channel_occupancy = np.sum(alpha, axis=0)
    busy_channels = np.sum(channel_occupancy > 1)
    total_channels = M
    avoidance_rate = ((total_channels - busy_channels) / total_channels) * 100 if total_channels > 0 else 0
    
    occupancy = np.mean(channel_occupancy) if total_channels > 0 else 0
    
    avg_throughput = total_throughput / served_sus if served_sus > 0 else 0.0
    qos_rate = (qos_satisfied / total_sus) * 100
    
    return {
        'served_sus': served_sus,
        'served_percentage': served_percentage,
        'allocated_channels': total_allocated_channels,
        'channel_rate': channel_allocation_rate,
        'total_throughput': total_throughput,
        'avg_throughput': avg_throughput,
        'qos_satisfied': qos_satisfied,
        'qos_rate': qos_rate,
        'occupancy': occupancy,
        'avoidance_rate': avoidance_rate,
        'time': elapsed_time
    }


def plot_convergence(history, save_path='convergence_v3.png'):
    """Génère et sauvegarde la courbe de convergence."""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(history, 'g-', linewidth=2)
        plt.xlabel('Itérations')
        plt.ylabel('Fitness')
        plt.title('Convergence de FOA V3 (Niching + Mémoire)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📊 Courbe sauvegardée: {save_path}")
    except ImportError:
        print("⚠️ Matplotlib non disponible")


def run_foa_v3(verbose=True):
    """Exécute FOA complet avec niching + mémoire (V3)."""
    
    np.random.seed(RANDOM_SEED)
    O = compute_overlap_matrix()
    population = initialize_population()
    avoidance = AvoidanceManager()
    
    if verbose:
        print("=" * 60)
        print("FOA V3 - Allocation spectrale CRN (Complet: Niching + Mémoire)")
        print("=" * 60)
        print(f"Population: {POPULATION}, Itérations: {ITERATIONS}")
    
    # Évaluation initiale (fitness brute)
    raw_fitnesses, allocations = evaluate_population(population, O, repair=True)
    
    # Enregistrer dans la mémoire
    for i, (pos, fit) in enumerate(zip(population, raw_fitnesses)):
        if fit > -np.inf:
            indices = avoidance.bloom.hash_allocation(allocations[i])
            avoidance.record_evaluation(indices, fit, fit)
    
    # Niching initial
    shared_fitnesses = compute_shared_fitness(population, raw_fitnesses)
    
    best_idx = np.argmax(shared_fitnesses)
    best_fitness = raw_fitnesses[best_idx]
    best_position = population[best_idx].copy()
    best_allocation = allocations[best_idx].copy()
    
    history = [best_fitness]
    
    if verbose:
        print(f"Meilleure fitness initiale (brute): {best_fitness:.4f}")
        print(f"Solutions valides: {sum(1 for f in raw_fitnesses if f > -np.inf)}/{POPULATION}")
        print("\nDébut de l'optimisation...")
    
    start_time = time.time()
    
    for iteration in range(ITERATIONS):
        new_population = []
        new_raw = []
        new_alloc = []
        
        for i in range(POPULATION):
            pos = population[i]
            alpha = allocations[i]
            fit = raw_fitnesses[i]
            
            indices = avoidance.bloom.hash_allocation(alpha)
            
            if avoidance.should_avoid(fit, best_fitness, indices):
                # Évitement
                new_pos = np.random.rand(N, M)
                new_fit, new_alpha = evaluate_position(new_pos, O, repair=True)
            else:
                # Déplacement normal
                new_pos = move_towards_best(pos, best_position)
                new_fit, new_alpha = evaluate_position(new_pos, O, repair=True)
            
            new_indices = avoidance.bloom.hash_allocation(new_alpha)
            avoidance.record_evaluation(new_indices, new_fit, best_fitness)
            
            new_population.append(new_pos)
            new_raw.append(new_fit)
            new_alloc.append(new_alpha)
        
        population = new_population
        raw_fitnesses = new_raw
        allocations = new_alloc
        
        # Niching
        new_shared = compute_shared_fitness(population, raw_fitnesses)
        
        best_shared_idx = np.argmax(new_shared)
        if raw_fitnesses[best_shared_idx] > best_fitness:
            best_fitness = raw_fitnesses[best_shared_idx]
            best_position = population[best_shared_idx].copy()
            best_allocation = allocations[best_shared_idx].copy()
            if verbose:
                print(f"✨ Itération {iteration}: nouvelle meilleure fitness = {best_fitness:.4f}")
        elif verbose and iteration % 20 == 0:
            print(f"   Itération {iteration}: fitness = {best_fitness:.4f}")
        
        history.append(best_fitness)
    
    elapsed_time = time.time() - start_time
    
    # Calcul des métriques
    metrics = compute_metrics(best_allocation, O, elapsed_time)
    
    if verbose:
        print("\n" + "=" * 60)
        print("OPTIMISATION TERMINÉE")
        print("=" * 60)
        print(f"Meilleure fitness: {best_fitness:.4f}")
        print(f"Temps: {elapsed_time:.2f} s")
        print(f"Canaux alloués: {np.sum(best_allocation)}")
        print(f"SUs servis: {metrics['served_sus']}/{N}")
        
        print("\n📊 MÉTRIQUES D'ÉVALUATION")
        print("=" * 60)
        print(f"Served_SUs: {metrics['served_sus']}")
        print(f"Occupancy: {metrics['occupancy']:.2f}")
        print(f"Throughput: {metrics['total_throughput']:.2f}")
        print(f"QoS_Rate: {metrics['qos_rate']:.2f}%")
        print(f"Avoidance_Rate: {metrics['avoidance_rate']:.2f}%")
        print(f"Time: {elapsed_time:.4f} s")
        
        avoidance.print_stats()
        
        # Courbe de convergence
        plot_convergence(history)
    
    stats = {
        'elapsed_time': elapsed_time,
        'metrics': metrics,
        'history': history
    }
    
    return best_fitness, best_allocation, history, stats


if __name__ == "__main__":
    run_foa_v3(verbose=True)