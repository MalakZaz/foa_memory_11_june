"""
run_v2.py - Exécutions multiples de V2 pour statistiques
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from config import RANDOM_SEED, N
from foa.foa_v2_bloom import run_foa_v2


def multiple_runs_v2(num_runs=30, verbose=True):
    """
    Exécute V2 plusieurs fois et affiche les statistiques.
    """
    fitnesses = []
    times = []
    served_list = []
    throughput_list = []
    qos_list = []
    avoidance_rates = []
    
    for run in range(num_runs):
        if verbose and (run + 1) % 5 == 0:
            print(f"Run {run+1}/{num_runs}...")
        
        np.random.seed(RANDOM_SEED + run)
        
        start = time.time()
        fit, _, _, stats = run_foa_v2(verbose=False)
        elapsed = time.time() - start
        
        # Extraction des métriques depuis stats
        metrics = stats.get('metrics', {})
        avoidance_stats = stats.get('avoidance_stats', {})
        
        fitnesses.append(fit)
        times.append(elapsed)
        served_list.append(metrics.get('served_sus', 0))
        throughput_list.append(metrics.get('total_throughput', 0))
        qos_list.append(metrics.get('qos_rate', 0))
        avoidance_rates.append(avoidance_stats.get('avoidance_rate', 0))
    
    # Statistiques
    print("\n" + "=" * 70)
    print(f"📊 STATISTIQUES V2 SUR {num_runs} RUNS")
    print("=" * 70)
    
    fitness_array = np.array(fitnesses)
    valid_fitnesses = fitness_array[fitness_array > -np.inf]
    
    print(f"\n--- FITNESS ---")
    print(f"  Moyenne : {np.mean(valid_fitnesses):.2f} ± {np.std(valid_fitnesses):.2f}")
    print(f"  Min : {np.min(valid_fitnesses):.2f}")
    print(f"  Max : {np.max(valid_fitnesses):.2f}")
    
    print(f"\n--- TEMPS D'EXÉCUTION ---")
    print(f"  Moyenne : {np.mean(times):.2f} ± {np.std(times):.2f} s")
    print(f"  Min : {np.min(times):.2f} s")
    print(f"  Max : {np.max(times):.2f} s")
    
    print(f"\n--- UTILISATEURS SERVIS ---")
    print(f"  Moyenne : {np.mean(served_list):.1f} / {N}")
    print(f"  Pourcentage : {np.mean(served_list)/N*100:.1f}%")
    
    print(f"\n--- THROUGHPUT ---")
    print(f"  Moyenne : {np.mean(throughput_list):.2f} ± {np.std(throughput_list):.2f} Mbps")
    
    print(f"\n--- QoS ---")
    print(f"  Taux satisfaction : {np.mean(qos_list):.1f}% ± {np.std(qos_list):.1f}%")
    
    print(f"\n--- ÉVITEMENT ---")
    print(f"  Taux moyen : {np.mean(avoidance_rates)*100:.1f}% ± {np.std(avoidance_rates)*100:.1f}%")
    
    print("\n" + "=" * 70)
    
    # Sauvegarde
    with open('results_v2_30runs.txt', 'w') as f:
        f.write(f"Résultats V2 - {num_runs} runs\n")
        f.write("=" * 50 + "\n")
        f.write(f"Fitness moyenne: {np.mean(valid_fitnesses):.4f} ± {np.std(valid_fitnesses):.4f}\n")
        f.write(f"Fitness min: {np.min(valid_fitnesses):.4f}\n")
        f.write(f"Fitness max: {np.max(valid_fitnesses):.4f}\n")
        f.write(f"Temps moyen: {np.mean(times):.2f} ± {np.std(times):.2f} s\n")
        f.write(f"SUs servis: {np.mean(served_list):.1f}/{N}\n")
        f.write(f"Throughput: {np.mean(throughput_list):.2f} Mbps\n")
        f.write(f"QoS: {np.mean(qos_list):.1f}%\n")
        f.write(f"Taux évitement: {np.mean(avoidance_rates)*100:.1f}%\n")
    
    print(f"\n✅ Résultats sauvegardés dans 'results_v2_30runs.txt'")
    
    return {
        'fitness_mean': np.mean(valid_fitnesses),
        'fitness_std': np.std(valid_fitnesses),
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'served_mean': np.mean(served_list),
        'throughput_mean': np.mean(throughput_list),
        'qos_mean': np.mean(qos_list),
        'avoidance_mean': np.mean(avoidance_rates)
    }


if __name__ == "__main__":
    results = multiple_runs_v2(num_runs=30, verbose=True)
    
    print("\n📈 RÉSUMÉ FINAL:")
    print(f"  Fitness: {results['fitness_mean']:.2f} ± {results['fitness_std']:.2f}")
    print(f"  Temps: {results['time_mean']:.2f} ± {results['time_std']:.2f} s")
    print(f"  SUs servis: {results['served_mean']:.1f}/{N}")
    print(f"  Taux évitement: {results['avoidance_mean']*100:.1f}%")