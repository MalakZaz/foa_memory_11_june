"""
run_v1.py - Test de V1 avec métriques complètes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime

# Importer V1
from foa.foa_v1_niching import run_foa_v1

# Paramètres de test (identiques à votre CSV)
TEST_CONFIG = {
    "scenario1": {"N": 20, "M": 10},
    "scenario2": {"N": 40, "M": 15},
    "scenario3": {"N": 60, "M": 20}
}

N_RUNS = 10  # 10 runs comme demandé


def test_v1_single_run(scenario_name, n, m, run_id):
    """Teste V1 sur une seule run et retourne les métriques"""
    
    print(f"\n🔄 Run {run_id}/{N_RUNS} sur {scenario_name} (N={n}, M={m})")
    
    # Modifier temporairement la config
    import config
    config.N = n
    config.M = m
    
    # Réinitialiser les caches dépendants de N et M
    import core.temporal
    import core.qos
    core.temporal.O = None  # Force recalcul
    core.qos.O = None
    if hasattr(core.qos, 'SINR_cache'):
        core.qos.SINR_cache = {}
    
    # Exécuter V1
    try:
        best_fitness, best_allocation, history, stats = run_foa_v1(verbose=False)
        
        # Extraire les métriques
        metrics = stats.get('metrics', {})
        
        result = {
            "Scenario": scenario_name,
            "N": n,
            "M": m,
            "Version": "V1",
            "Run": run_id,
            "Fitness": best_fitness,
            "Time": metrics.get('time', 0),
            "Served_SUs": metrics.get('served_sus', 0),
            "Occupancy": metrics.get('occupancy', 0),
            "Throughput": metrics.get('total_throughput', 0),
            "QoS_Rate": metrics.get('qos_rate', 0),
            "Avoidance_Rate": metrics.get('avoidance_rate', 0)
        }
        
        print(f"   ✅ Fitness: {result['Fitness']:.2f}, Served_SUs: {result['Served_SUs']}, QoS: {result['QoS_Rate']:.1f}%")
        return result
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return {
            "Scenario": scenario_name,
            "N": n,
            "M": m,
            "Version": "V1",
            "Run": run_id,
            "Fitness": -1,
            "Time": 0,
            "Served_SUs": 0,
            "Occupancy": 0,
            "Throughput": 0,
            "QoS_Rate": 0,
            "Avoidance_Rate": 0
        }


def main():
    print("=" * 60)
    print("🧪 TEST DE V1 - 10 RUNS PAR SCÉNARIO")
    print("=" * 60)
    
    all_results = []
    
    for scenario_name, params in TEST_CONFIG.items():
        print(f"\n📂 Scénario: {scenario_name} (N={params['N']}, M={params['M']})")
        print("-" * 40)
        
        for run in range(1, N_RUNS + 1):
            result = test_v1_single_run(scenario_name, params['N'], params['M'], run)
            all_results.append(result)
    
    # Sauvegarde des résultats
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"v1_test_results_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES RÉSULTATS V1")
    print("=" * 60)
    
    for scenario_name in TEST_CONFIG.keys():
        scenario_df = df[df["Scenario"] == scenario_name]
        print(f"\n📍 {scenario_name}:")
        print(f"   Fitness:       {scenario_df['Fitness'].mean():.2f} ± {scenario_df['Fitness'].std():.3f}")
        print(f"   Served_SUs:    {scenario_df['Served_SUs'].mean():.1f} ± {scenario_df['Served_SUs'].std():.3f}")
        print(f"   QoS_Rate:      {scenario_df['QoS_Rate'].mean():.1f} ± {scenario_df['QoS_Rate'].std():.3f}%")
        print(f"   Avoidance_Rate:{scenario_df['Avoidance_Rate'].mean():.1f} ± {scenario_df['Avoidance_Rate'].std():.3f}%")
        print(f"   Time:          {scenario_df['Time'].mean():.3f} ± {scenario_df['Time'].std():.3f} s")
    
    print(f"\n✅ Fichier sauvegardé: {output_file}")
    
    # Vérification rapide
    total_served = df['Served_SUs'].sum()
    print(f"\n🎯 VÉRIFICATION: Total SUs servis sur {len(df)} runs = {total_served}")
    if total_served == 0:
        print("⚠️ PROBLÈME: Aucun SU servi ! V1 ne fonctionne pas encore correctement.")
    else:
        print("✅ BON SIGNE: V1 sert des SUs !")
    
    return df


if __name__ == "__main__":
    # Sauvegarder la config originale
    import config
    original_N = config.N
    original_M = config.M
    
    try:
        results = main()
    finally:
        # Restaurer config originale
        config.N = original_N
        config.M = original_M