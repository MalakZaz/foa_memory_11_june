"""
generate_tables.py - Génération des tableaux LaTeX pour l'article
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import N, M, POPULATION, ITERATIONS, ETA, R_MIN, SINR_MIN, P_TX, NOISE_POWER, DELTA_T


def generate_table_parameters():
    """Tableau 1: Paramètres de simulation (format LaTeX)"""
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Paramètres de simulation}}
\\label{{tab:parameters}}
\\begin{{tabular}}{{ll}}
\\toprule
Paramètre & Valeur \\\\
\\midrule
Nombre de SUs ($N$) & {N} \\\\
Nombre de canaux ($M$) & {M} \\\\
Taille de population & {POPULATION} \\\\
Itérations & {ITERATIONS} \\\\
Facteur d'exploration ($\\eta$) & {ETA} \\\\
Débit minimum ($R_{{\\min}}$) & {R_MIN} Mbps \\\\
SINR minimum ($\\gamma_{{\\min}}$) & {SINR_MIN} \\\\
Puissance d'émission ($P_{{TX}}$) & {P_TX} W \\\\
Bruit thermique ($\\sigma^2$) & {NOISE_POWER} W \\\\
Fenêtre temporelle ($\\Delta t$) & {DELTA_T:.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def generate_table_statistics(results):
    """
    Tableau 2: Statistiques sur 30 runs.
    results: dict avec 'fitness_mean', 'fitness_std', 'time_mean', etc.
    """
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Statistiques sur 30 exécutions indépendantes}}
\\label{{tab:statistics}}
\\begin{{tabular}}{{lccc}}
\\toprule
Métrique & Moyenne & Écart-type & Min / Max \\\\
\\midrule
Fitness & {results['fitness_mean']:.2f} & {results['fitness_std']:.2f} & - \\\\
Temps d'exécution (s) & {results['time_mean']:.2f} & {results['time_std']:.2f} & - \\\\
SUs servis & {results['served_mean']:.1f} & - & - \\\\
Throughput total (Mbps) & {results['throughput_mean']:.1f} & - & - \\\\
Taux de satisfaction QoS (\\%) & {results['qos_mean']:.1f} & - & - \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def generate_table_scalability(scalability_results):
    """
    Tableau 3: Étude de scalabilité.
    scalability_results: dict {N: {'fitness': ..., 'time': ..., 'served': ..., 'qos': ...}}
    """
    rows = ""
    for N, res in scalability_results.items():
        rows += f"{N} & {res['fitness']:.2f} $\\pm$ {res['fitness_std']:.2f} & "
        rows += f"{res['time']:.2f} $\\pm$ {res['time_std']:.2f} & "
        rows += f"{res['served']:.1f}/{N} & {res['qos']:.1f}\\% \\\\\n"
    
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Étude de scalabilité - Impact du nombre de SUs}}
\\label{{tab:scalability}}
\\begin{{tabular}}{{lcccc}}
\\toprule
N & Fitness & Temps (s) & SUs servis & QoS (\\%) \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def save_latex_tables(results, scalability_results):
    """Sauvegarde tous les tableaux dans des fichiers .tex"""
    
    with open('table_parameters.tex', 'w', encoding='utf-8') as f:
        f.write(generate_table_parameters())
    print("✅ table_parameters.tex")
    
    with open('table_statistics.tex', 'w', encoding='utf-8') as f:
        f.write(generate_table_statistics(results))
    print("✅ table_statistics.tex")
    
    with open('table_scalability.tex', 'w', encoding='utf-8') as f:
        f.write(generate_table_scalability(scalability_results))
    print("✅ table_scalability.tex")


if __name__ == "__main__":
    print("=" * 60)
    print("GÉNÉRATION DES TABLEAUX LATEX")
    print("=" * 60)
    
    # Exemple avec des résultats factices
    example_results = {
        'fitness_mean': 1250.45,
        'fitness_std': 45.2,
        'time_mean': 28.5,
        'time_std': 3.2,
        'served_mean': 24.5,
        'throughput_mean': 850.6,
        'qos_mean': 78.3
    }
    
    example_scalability = {
        20: {'fitness': 450.2, 'fitness_std': 12.3, 'time': 8.5, 'time_std': 1.2, 'served': 14, 'qos': 70.0},
        40: {'fitness': 850.5, 'fitness_std': 18.5, 'time': 22.3, 'time_std': 2.5, 'served': 25, 'qos': 72.0},
        60: {'fitness': 1050.3, 'fitness_std': 22.1, 'time': 45.2, 'time_std': 3.8, 'served': 32, 'qos': 68.0},
        80: {'fitness': 1180.5, 'fitness_std': 28.4, 'time': 78.5, 'time_std': 5.2, 'served': 38, 'qos': 65.0},
        100: {'fitness': 1250.4, 'fitness_std': 32.5, 'time': 120.5, 'time_std': 8.5, 'served': 42, 'qos': 60.0}
    }
    
    save_latex_tables(example_results, example_scalability)
    
    print("\n" + "=" * 60)
    print("✅ Tableaux générés avec succès!")
    print("=" * 60)
    print("\nFichiers créés:")
    print("  - table_parameters.tex")
    print("  - table_statistics.tex")
    print("  - table_scalability.tex")