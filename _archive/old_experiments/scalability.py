"""
scalability.py - Étude de scalabilité (N variable)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import importlib
from config import M, RANDOM_SEED
from foa.foa_v0 import run_foa_v0


def scalability_study(N_values=[20, 40, 60, 80, 100], num_runs=5):
    """
    Étudie l'impact du nombre de SUs sur les performances.
    """
    results = {}
    
    print("=" * 80)
    print("ÉTUDE DE SCALABILITÉ")
    print("=" * 80)
    print(f"M fixe = {M}, {num_runs} runs par configuration")
    print("-" * 80)
    
    for N in N_values:
        print(f"\n--- N = {N} ---")
        
        # Modifier dynamiquement N
        import config
        config.N = N
        importlib.reload(config)
        
        # Réinitialiser les données
        config.b_s = np.random.randint(4, 10, N)
        config.t_start = np.random.uniform(0, 20, N)
        config.d_s = np.random.randint(2, 10, N)
        config.w_s = np.ones(N)
        config.gamma = np.random.uniform(0.5, 3.0, (N, M))
        config.A = np.random.choice([0, 1], size=(N, M), p=[0.3, 0.7])
        config.h = np.random.uniform(0.5, 2.0, (N, M))
        config.DELTA_T = np.max(config.d_s)
        
        fitnesses = []
        times = []
        served = []
        throughputs = []
        qos = []
        
        for run in range(num_runs):
            np.random.seed(RANDOM_SEED + run)
            fit, _, _, stats = run_foa_v0(verbose=False)
            
            metrics = stats.get('metrics', {})
            fitnesses.append(fit)
            times.append(stats['elapsed_time'])
            served.append(metrics.get('served_sus', 0))
            throughputs.append(metrics.get('total_throughput', 0))
            qos.append(metrics.get('qos_rate', 0))
        
        results[N] = {
            'fitness': np.mean(fitnesses),
            'fitness_std': np.std(fitnesses),
            'time': np.mean(times),
            'time_std': np.std(times),
            'served': np.mean(served),
            'throughput': np.mean(throughputs),
            'qos': np.mean(qos)
        }
    
    # Affichage du tableau
    print("\n" + "=" * 100)
    print("RÉSULTATS DE L'ÉTUDE DE SCALABILITÉ")
    print("=" * 100)
    print(f"{'N':<8} {'Fitness':<18} {'Temps (s)':<14} {'SUs servis':<14} {'Throughput':<14} {'QoS (%)':<10}")
    print("-" * 100)
    
    for N, res in results.items():
        print(f"{N:<8} {res['fitness']:.2f} ± {res['fitness_std']:.2f}   "
              f"{res['time']:.2f} ± {res['time_std']:.2f}       "
              f"{res['served']:.1f}/{N:<5}      "
              f"{res['throughput']:.1f}        "
              f"{res['qos']:.1f}%")
    
    print("=" * 100)
    
    # Sauvegarde
    with open('scalability_results.txt', 'w') as f:
        f.write("ÉTUDE DE SCALABILITÉ\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'N':<8} {'Fitness':<18} {'Temps (s)':<14} {'SUs servis':<14} {'QoS (%)':<10}\n")
        for N, res in results.items():
            f.write(f"{N:<8} {res['fitness']:.2f} ± {res['fitness_std']:.2f}   "
                    f"{res['time']:.2f} ± {res['time_std']:.2f}       "
                    f"{res['served']:.1f}/{N:<5}      "
                    f"{res['qos']:.1f}%\n")
    
    print("\n✅ Résultats sauvegardés dans 'scalability_results.txt'")
    
    return results


if __name__ == "__main__":
    scalability_study(num_runs=5)