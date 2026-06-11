"""
compare_all_versions.py - Exécute et compare V0, V1, V2, V3 (30 runs chacun)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from config import N, M, RANDOM_SEED

# Import des versions
from foa.foa_v0 import run_foa_v0
from foa.foa_v1_niching import run_foa_v1
from foa.foa_v2_bloom import run_foa_v2
from foa.foa_v3_complete import run_foa_v3


def run_version(name, func, num_runs=30, verbose=False):
    """Exécute une version N fois et retourne les statistiques."""
    print(f"\n--- {name} ({num_runs} runs) ---")
    
    fitnesses = []
    times = []
    served_list = []
    throughput_list = []
    qos_list = []
    
    for run in range(num_runs):
        if verbose and run % 10 == 0:
            print(f"  Run {run+1}/{num_runs}...")
        
        # Changer la graine à chaque run
        np.random.seed(RANDOM_SEED + run)
        
        start = time.time()
        fitness, allocation, history, stats = func(verbose=False, show_params=False)
        elapsed = time.time() - start
        
        metrics = stats.get('metrics', {})
        
        fitnesses.append(fitness)
        times.append(elapsed)
        served_list.append(metrics.get('served_sus', np.sum(np.sum(allocation, axis=1) > 0)))
        throughput_list.append(metrics.get('total_throughput', 0))
        qos_list.append(metrics.get('qos_satisfied_users', 0))
    
    # Calcul des statistiques
    fitness_array = np.array(fitnesses)
    valid_fitnesses = fitness_array[fitness_array > -np.inf]
    
    return {
        'name': name,
        'num_runs': num_runs,
        'valid_runs': len(valid_fitnesses),
        'fitness_mean': np.mean(valid_fitnesses) if len(valid_fitnesses) > 0 else -np.inf,
        'fitness_std': np.std(valid_fitnesses) if len(valid_fitnesses) > 0 else 0,
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'served_mean': np.mean(served_list),
        'served_std': np.std(served_list),
        'throughput_mean': np.mean(throughput_list),
        'throughput_std': np.std(throughput_list),
        'qos_mean': np.mean(qos_list),
        'qos_std': np.std(qos_list)
    }


def main(num_runs=30, verbose=True):
    print("=" * 80)
    print(f"COMPARAISON DES VERSIONS FOA (N={N}, M={M})")
    print(f"Nombre de runs par version : {num_runs}")
    print("=" * 80)
    
    results = []
    
    # Exécution des 4 versions
    results.append(run_version("V0 - FOA Standard", run_foa_v0, num_runs, verbose))
    results.append(run_version("V1 - FOA + Niching", run_foa_v1, num_runs, verbose))
    results.append(run_version("V2 - FOA + Bloom Memory", run_foa_v2, num_runs, verbose))
    results.append(run_version("V3 - FOA Complet", run_foa_v3, num_runs, verbose))
    
    # Affichage du tableau
    print("\n" + "=" * 110)
    print("📊 TABLEAU COMPARATIF DES RÉSULTATS (30 runs)")
    print("=" * 110)
    print(f"{'Version':<22} {'Fitness':<20} {'SUs servis':<15} {'Throughput (Mbps)':<20} {'Temps (s)':<15} {'QoS':<10}")
    print("-" * 110)
    
    for r in results:
        print(f"{r['name']:<22} {r['fitness_mean']:.2f} ± {r['fitness_std']:.2f}   "
              f"{r['served_mean']:.1f} ± {r['served_std']:.1f}        "
              f"{r['throughput_mean']:.1f} ± {r['throughput_std']:.1f}     "
              f"{r['time_mean']:.2f} ± {r['time_std']:.2f}   "
              f"{r['qos_mean']:.1f} ± {r['qos_std']:.1f}")
    
    print("=" * 110)
    
    # Sauvegarde des résultats
    with open('comparison_results_30runs.txt', 'w') as f:
        f.write(f"COMPARAISON DES VERSIONS FOA (N={N}, M={M})\n")
        f.write(f"Nombre de runs par version : {num_runs}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Version':<22} {'Fitness':<20} {'SUs servis':<15} {'Temps (s)':<15}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r['name']:<22} {r['fitness_mean']:.2f} ± {r['fitness_std']:.2f}   "
                    f"{r['served_mean']:.1f} ± {r['served_std']:.1f}             "
                    f"{r['time_mean']:.2f} ± {r['time_std']:.2f}\n")
    
    print("\n✅ Résultats sauvegardés dans 'comparison_results_30runs.txt'")


if __name__ == "__main__":
    # 30 runs pour statistiques
    main(num_runs=1, verbose=True)