"""
test_v3_only.py
Teste uniquement la version V3 sur 10 runs pour chaque scénario
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
VERSIONS_A_TESTER = ["V3"]
N_RUNS = 10

# Paramètres des scénarios (identiques au CSV)
SCENARIOS = {
    "scenario1": {"N": 20, "M": 10},
    "scenario2": {"N": 40, "M": 15},
    "scenario3": {"N": 60, "M": 20}
}

# Résultats attendus depuis votre CSV (valeurs de référence)
# V3 dans votre CSV a: Fitness=794.5250, Time~4.15-4.22
V3_REFERENCE = {
    "Fitness": 794.5250,
    "Time_mean": 4.18,
    "Time_std": 0.02,
    "Served_SUs": 0,
    "Occupancy": 0,
    "Throughput": 0,
    "QoS_Rate": 0,
    "Avoidance_Rate": 0
}


def run_simulation(scenario_name, n, m, version, run_id):
    """
    ⚠️ À MODIFIER : Remplacez cette fonction par votre vraie simulation
    
    Actuellement, cette fonction génère des données cohérentes avec votre CSV
    mais vous devez la remplacer par votre vrai code.
    """
    
    print(f"  Run {run_id}/{N_RUNS}...")
    
    # -----------------------------------------------------------------
    # 🔧 REMPLACEZ À PARTIR D'ICI PAR VOTRE VRAIE SIMULATION
    # -----------------------------------------------------------------
    
    # Génération de données réalistes basées sur votre CSV
    # (À SUPPRIMER quand vous aurez votre vraie simulation)
    np.random.seed(run_id * 100)  # Pour reproductibilité
    
    # Fitness stable comme dans votre CSV
    fitness = V3_REFERENCE["Fitness"] + np.random.normal(0, 0.3)
    
    # Time entre 4.15 et 4.22 comme dans votre CSV
    time = V3_REFERENCE["Time_mean"] + np.random.normal(0, V3_REFERENCE["Time_std"])
    
    # Votre CSV montre que V3 sert 0 SU
    served_sus = V3_REFERENCE["Served_SUs"]
    occupancy = V3_REFERENCE["Occupancy"]
    throughput = V3_REFERENCE["Throughput"]
    qos_rate = V3_REFERENCE["QoS_Rate"]
    avoidance_rate = V3_REFERENCE["Avoidance_Rate"]
    
    # -----------------------------------------------------------------
    # FIN DE LA PARTIE À REMPLACER
    # -----------------------------------------------------------------
    
    return {
        "Fitness": fitness,
        "Time": time,
        "Served_SUs": served_sus,
        "Occupancy": occupancy,
        "Throughput": throughput,
        "QoS_Rate": qos_rate,
        "Avoidance_Rate": avoidance_rate
    }


def main():
    all_results = []
    
    print("=" * 60)
    print("TEST V3 UNIQUEMENT - 10 RUNS PAR SCÉNARIO")
    print("=" * 60)
    
    for scenario_name, params in SCENARIOS.items():
        print(f"\n📊 Scénario: {scenario_name} (N={params['N']}, M={params['M']})")
        print("-" * 40)
        
        for version in VERSIONS_A_TESTER:
            print(f"\n🎮 Version: {version}")
            
            for run in range(1, N_RUNS + 1):
                result = run_simulation(
                    scenario_name, 
                    params['N'], 
                    params['M'], 
                    version, 
                    run
                )
                
                all_results.append({
                    "Scenario": scenario_name,
                    "N": params['N'],
                    "M": params['M'],
                    "Version": version,
                    "Run": run,
                    "Fitness": result["Fitness"],
                    "Time": result["Time"],
                    "Served_SUs": result["Served_SUs"],
                    "Occupancy": result["Occupancy"],
                    "Throughput": result["Throughput"],
                    "QoS_Rate": result["QoS_Rate"],
                    "Avoidance_Rate": result["Avoidance_Rate"]
                })
    
    # Sauvegarde des résultats
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"v3_only_results_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    
    # Affichage des statistiques
    print("\n" + "=" * 60)
    print("📈 RÉSUMÉ DES RÉSULTATS V3")
    print("=" * 60)
    
    for scenario_name in SCENARIOS.keys():
        scenario_df = df[df["Scenario"] == scenario_name]
        print(f"\n📍 {scenario_name}:")
        print(f"   Fitness: {scenario_df['Fitness'].mean():.2f} ± {scenario_df['Fitness'].std():.3f}")
        print(f"   Time:    {scenario_df['Time'].mean():.3f} ± {scenario_df['Time'].std():.3f} s")
        print(f"   Served_SUs: {scenario_df['Served_SUs'].mean():.0f}")
    
    print(f"\n✅ Fichier sauvegardé: {output_file}")
    
    # Comparaison avec les valeurs du CSV original
    print("\n" + "=" * 60)
    print("📊 COMPARAISON AVEC VOTRE CSV ORIGINAL")
    print("=" * 60)
    print("Valeurs attendues pour V3 d'après votre CSV:")
    print(f"   Fitness: 794.5250")
    print(f"   Time:    ~4.15-4.22 s")
    print(f"   Served_SUs: 0")
    
    return df


# Script d'analyse supplémentaire
def analyze_results(csv_file):
    """Analyse les résultats sauvegardés"""
    df = pd.read_csv(csv_file)
    
    print("\n" + "=" * 60)
    print("🔬 ANALYSE DÉTAILLÉE")
    print("=" * 60)
    
    # Boxplots par scénario
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Fitness par scénario
        df.boxplot(column="Fitness", by="Scenario", ax=axes[0])
        axes[0].set_title("Fitness par Scénario")
        axes[0].set_ylabel("Fitness")
        axes[0].grid(True)
        
        # Time par scénario
        df.boxplot(column="Time", by="Scenario", ax=axes[1])
        axes[1].set_title("Temps d'exécution par Scénario")
        axes[1].set_ylabel("Time (s)")
        axes[1].grid(True)
        
        plt.suptitle("Analyse V3 - 10 runs")
        plt.tight_layout()
        plt.savefig("v3_analysis.png", dpi=150)
        print("\n✅ Graphique sauvegardé: v3_analysis.png")
        plt.show()
    except ImportError:
        print("\n⚠️ matplotlib non installé - installez-le pour les graphiques")
    
    return df


if __name__ == "__main__":
    # Exécuter les simulations
    results_df = main()
    
    # Demander si analyse graphique
    print("\n📊 Voulez-vous générer l'analyse graphique ? (o/n)")
    if input().lower() == 'o':
        analyze_results("v3_only_results_*.csv")  # Remplacez par le nom exact