"""
run_scenario1_10runs.py
Exécute uniquement le Scenario 1 avec 10 runs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import csv
import pandas as pd
from datetime import datetime
from config import RANDOM_SEED
from foa.foa_v0 import run_foa_v0
from foa.foa_v1_niching import run_foa_v1
from foa.foa_v2_bloom import run_foa_v2
from foa.foa_v3_complete import run_foa_v3

# ============================================
# CONFIGURATION - SCENARIO 1 UNIQUEMENT
# ============================================
SCENARIO = {'N': 20, 'M': 10, 'name': 'scenario1'}

VERSIONS = [
    {'name': 'V0', 'func': run_foa_v0},
    {'name': 'V1', 'func': run_foa_v1},
    {'name': 'V2', 'func': run_foa_v2},
    {'name': 'V3', 'func': run_foa_v3}
]

NUM_RUNS = 10  # 10 runs seulement

# ============================================
# FONCTION PRINCIPALE
# ============================================
def main():
    print("=" * 70)
    print(f"🎯 EXÉCUTION - {SCENARIO['name'].upper()} (N={SCENARIO['N']}, M={SCENARIO['M']})")
    print("=" * 70)
    print(f"Versions: {[v['name'] for v in VERSIONS]}")
    print(f"Nombre de runs: {NUM_RUNS}")
    print("=" * 70)
    
    # Créer le dossier results
    os.makedirs('results', exist_ok=True)
    
    # Nom du fichier de sortie avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f'results/scenario1_10runs_{timestamp}.csv'
    
    print(f"\n📁 Fichier de sortie : {csv_path}\n")
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Scenario', 'N', 'M', 'Version', 'Run', 'Fitness', 
                         'Time', 'Served_SUs', 'Occupancy', 'Throughput', 
                         'QoS_Rate', 'Avoidance_Rate'])
        
        for version in VERSIONS:
            print(f"\n--- {version['name']} ---")
            
            for run in range(1, NUM_RUNS + 1):
                np.random.seed(RANDOM_SEED + run)
                
                # Modifier la config pour ce run
                import config
                config.N = SCENARIO['N']
                config.M = SCENARIO['M']
                
                # Recharger et réinitialiser les caches
                import importlib
                import core.temporal
                import core.qos
                core.temporal.O = None
                core.qos.O = None
                if hasattr(core.qos, 'SINR_cache'):
                    core.qos.SINR_cache = {}
                importlib.reload(config)
                
                print(f"  Run {run:2d}/{NUM_RUNS}...", end=' ', flush=True)
                
                start = time.time()
                fitness, _, _, stats = version['func'](verbose=False)
                elapsed = time.time() - start
                
                metrics = stats.get('metrics', {})
                avoidance_stats = stats.get('avoidance_stats', {})
                
                served = metrics.get('served_sus', 0)
                occupancy = metrics.get('channel_rate', 0)
                throughput = metrics.get('total_throughput', 0)
                qos_rate = metrics.get('qos_rate', 0)
                avoidance_rate = avoidance_stats.get('avoidance_rate', 0) * 100
                
                writer.writerow([
                    SCENARIO['name'], SCENARIO['N'], SCENARIO['M'],
                    version['name'], run,
                    f"{fitness:.4f}", f"{elapsed:.4f}",
                    served, f"{occupancy:.2f}", f"{throughput:.2f}",
                    f"{qos_rate:.2f}", f"{avoidance_rate:.2f}"
                ])
                
                print(f"Fitness={fitness:.2f}, Served={served}, QoS={qos_rate:.1f}%, Temps={elapsed:.2f}s")
    
    print(f"\n✅ Résultats sauvegardés : {csv_path}")
    
    # Afficher un résumé
    df = pd.read_csv(csv_path)
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DES RÉSULTATS (moyennes sur 10 runs)")
    print("=" * 70)
    
    summary = df.groupby('Version').agg({
        'Fitness': ['mean', 'std'],
        'Served_SUs': 'mean',
        'QoS_Rate': 'mean',
        'Time': 'mean',
        'Avoidance_Rate': 'mean'
    }).round(2)
    
    print(summary)
    
    print("\n✅ Script terminé !")
    return csv_path

# ============================================
# GÉNÉRATION DU GRAPHIQUE APRÈS EXÉCUTION
# ============================================
def generate_graph(csv_path):
    """Génère automatiquement le graphique après l'exécution"""
    try:
        import matplotlib.pyplot as plt
        
        df = pd.read_csv(csv_path)
        
        # Créer le dossier figures
        os.makedirs('figures', exist_ok=True)
        
        # Style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'
        
        # Couleurs
        colors = {'V0': '#1f77b4', 'V1': '#ff7f0e', 'V2': '#2ca02c', 'V3': '#d62728'}
        markers = {'V0': 'o', 'V1': 's', 'V2': '^', 'V3': 'D'}
        
        plt.figure(figsize=(10, 6))
        
        for version in ['V0', 'V1', 'V2', 'V3']:
            version_df = df[df['Version'] == version].sort_values('Run')
            
            if len(version_df) > 0:
                fitness_values = version_df['Fitness'].values
                runs = range(1, len(fitness_values) + 1)
                mean_fitness = np.cumsum(fitness_values) / np.arange(1, len(fitness_values) + 1)
                
                plt.plot(runs, mean_fitness, 
                        color=colors[version],
                        marker=markers[version],
                        markevery=2,
                        markersize=6,
                        linewidth=2,
                        label=f'{version} (final: {mean_fitness[-1]:.2f})')
        
        plt.xlabel('Run Number', fontsize=12)
        plt.ylabel('Mean Best Fitness Value', fontsize=12)
        plt.title('Mean Best Fitness Value for Scenario 1 (10 runs)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 11)
        
        plt.tight_layout()
        
        output_path = 'figures/mean_fitness_scenario1_10runs.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Graphique généré : {output_path}")
        
    except ImportError:
        print("\n⚠️ Matplotlib non disponible, graphique non généré")

# ============================================
# EXÉCUTION
# ============================================
if __name__ == "__main__":
    csv_file = main()
    generate_graph(csv_file)