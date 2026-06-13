"""
exact_optimization.py
Calcul de l'optimum exact du problème d'allocation spectrale
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import matplotlib.pyplot as plt

# Configuration
SCENARIOS = [
    {'N': 20, 'M': 10, 'name': 'scenario1'},
    {'N': 40, 'M': 15, 'name': 'scenario2'},
    {'N': 60, 'M': 20, 'name': 'scenario3'}
]


def compute_overlap_matrix(N, M):
    """Calcule la matrice de chevauchement"""
    np.random.seed(42)
    O = np.random.rand(N, M) < 0.3
    return O.astype(int)


def solve_exact_milp(N, M, O, time_limit=300):
    """
    Résolution exacte par Programmation Linéaire Mixte en Nombres Entiers
    Maximise: Σ (20 * served_su + throughput)
    """
    try:
        from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, lpSum, PULP_CBC_CMD
    except ImportError:
        print("  Installation de pulp...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pulp"])
        from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, lpSum, PULP_CBC_CMD
    
    # Créer le problème
    prob = LpProblem("Allocation_Spectrale", LpMaximize)
    
    # Variables: x[s][p] = 1 si SU s utilise canal p
    x = {}
    for s in range(N):
        for p in range(M):
            x[s, p] = LpVariable(f"x_{s}_{p}", 0, 1, LpBinary)
    
    # Variable y[s] = 1 si SU s est servi
    y = {}
    for s in range(N):
        y[s] = LpVariable(f"y_{s}", 0, 1, LpBinary)
    
    # Objectif: maximiser le nombre de SUs servis + QoS
    # Fitness = 20 * nombre_SUs_servis + somme_des_debits
    obj_terms = []
    
    # Partie 1: récompenser les SUs servis
    for s in range(N):
        obj_terms.append(20 * y[s])
    
    # Partie 2: récompenser les débits (QoS)
    # Calcul simplifié du débit
    B_p = 180  # kHz
    for s in range(N):
        for p in range(M):
            # SINR simplifié: 1 / (interférence potentielle + 0.01)
            interference = sum(1 for s2 in range(N) if s2 != s)
            sinr = 1.0 / (interference + 0.01)
            rate = B_p * np.log2(1 + sinr)
            obj_terms.append((rate / 100) * x[s, p])  # Divisé par 100 pour échelle
    
    prob += lpSum(obj_terms)
    
    # Contrainte 1: Un SU ne peut utiliser qu'un seul canal
    for s in range(N):
        prob += lpSum(x[s, p] for p in range(M)) <= 1
    
    # Contrainte 2: Lien entre x et y (y[s] = 1 si au moins un x[s,p] = 1)
    for s in range(N):
        prob += y[s] <= lpSum(x[s, p] for p in range(M))
        prob += y[s] >= lpSum(x[s, p] for p in range(M)) / M
    
    # Contrainte 3: Pas de collision (deux SUs qui se chevauchent ne peuvent pas utiliser le même canal)
    for p in range(M):
        for s1 in range(N):
            for s2 in range(s1 + 1, N):
                if O[s1, p] and O[s2, p]:
                    prob += x[s1, p] + x[s2, p] <= 1
    
    print(f"  Variables: {N * M + N}")
    print(f"  Contraintes: {len(prob.constraints)}")
    print("  Résolution MILP en cours...")
    
    start_time = time.time()
    
    # Résoudre avec CBC (solveur gratuit inclus dans pulp)
    solver = PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    prob.solve(solver)
    
    elapsed = time.time() - start_time
    
    if prob.status == 1:  # Optimal
        # Extraire la solution
        allocation = np.zeros((N, M))
        for s in range(N):
            for p in range(M):
                if x[s, p].varValue > 0.5:
                    allocation[s, p] = 1
        
        served_sus = sum(1 for s in range(N) if y[s].varValue > 0.5)
        
        # Calculer le débit total
        total_throughput = 0
        qos_count = 0
        for s in range(N):
            for p in range(M):
                if allocation[s, p] == 1:
                    interference = sum(1 for s2 in range(N) if s2 != s and O[s2, p])
                    sinr = 1.0 / (interference + 0.01)
                    rate = B_p * np.log2(1 + sinr)
                    total_throughput += rate
                    if sinr >= 10:
                        qos_count += 1
        
        qos_rate = (qos_count / N) * 100 if N > 0 else 0
        fitness = served_sus * 20 + total_throughput / 100
        
        metrics = {
            'fitness': fitness,
            'served_sus': served_sus,
            'total_throughput': total_throughput,
            'qos_rate': qos_rate,
            'time': elapsed,
            'status': 'Optimal',
            'allocation': allocation
        }
    else:
        metrics = {
            'fitness': -1,
            'served_sus': 0,
            'total_throughput': 0,
            'qos_rate': 0,
            'time': elapsed,
            'status': f'Non optimal (status={prob.status})'
        }
    
    return metrics


def solve_exact_bruteforce_small(N, M, O, max_combinations=10**6):
    """
    Force brute pour petites instances (N <= 12)
    """
    from itertools import product
    
    n_allocations = (M + 1) ** N
    if n_allocations > max_combinations:
        return {'fitness': -1, 'status': f'Trop grand: {n_allocations} combinaisons (> {max_combinations})'}
    
    print(f"  Nombre de combinaisons: {n_allocations}")
    
    best_fitness = -np.inf
    best_allocation = None
    
    B_p = 180
    
    start_time = time.time()
    
    for channels in product(range(M + 1), repeat=N):
        # Vérifier collisions
        valid = True
        for p in range(M):
            sus_on_p = [s for s, ch in enumerate(channels) if ch == p]
            for i, s1 in enumerate(sus_on_p):
                for s2 in sus_on_p[i+1:]:
                    if O[s1, p] and O[s2, p]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Calculer fitness
        served = sum(1 for ch in channels if ch < M)
        throughput = 0
        qos = 0
        
        for s, p in enumerate(channels):
            if p < M:
                interference = sum(1 for s2 in range(N) if s2 != s and channels[s2] == p and O[s2, p])
                sinr = 1.0 / (interference + 0.01)
                rate = B_p * np.log2(1 + sinr)
                throughput += rate
                if sinr >= 10:
                    qos += 1
        
        fitness = served * 20 + throughput / 100
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_allocation = channels
    
    elapsed = time.time() - start_time
    
    return {
        'fitness': best_fitness,
        'served_sus': sum(1 for ch in best_allocation if ch < M) if best_allocation else 0,
        'total_throughput': throughput if best_allocation else 0,
        'qos_rate': (qos / N) * 100 if best_allocation else 0,
        'time': elapsed,
        'status': 'Optimal (bruteforce)'
    }


def run_exact_optimization():
    """Exécute l'optimisation exacte"""
    
    print("=" * 70)
    print("🎯 OPTIMISATION EXACTE - BORNE SUPÉRIEURE THÉORIQUE")
    print("=" * 70)
    
    results = {}
    
    for scenario in SCENARIOS:
        print(f"\n📊 Scénario: {scenario['name']} (N={scenario['N']}, M={scenario['M']})")
        print("-" * 50)
        
        O = compute_overlap_matrix(scenario['N'], scenario['M'])
        
        # MILP
        print("\n  🧮 Méthode MILP (Branch & Bound):")
        milp_results = solve_exact_milp(scenario['N'], scenario['M'], O)
        
        if milp_results['fitness'] > 0:
            print(f"    ✓ Fitness optimale: {milp_results['fitness']:.2f}")
            print(f"    ✓ SUs servis: {milp_results['served_sus']}/{scenario['N']}")
            print(f"    ✓ QoS Rate: {milp_results['qos_rate']:.1f}%")
            print(f"    ✓ Temps: {milp_results['time']:.2f}s")
        else:
            print(f"    ⚠️ Problème: {milp_results['status']}")
        
        results[scenario['name']] = milp_results
    
    return results


def main():
    """Fonction principale"""
    
    print("\n" + "=" * 70)
    print("🔧 Installation des dépendances si nécessaire...")
    print("=" * 70)
    
    # Installer pulp
    try:
        import pulp
        print("✓ pulp déjà installé")
    except ImportError:
        print("Installation de pulp...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pulp"])
        print("✓ pulp installé")
    
    # Exécuter
    results = run_exact_optimization()
    
    # Afficher le résumé
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DES OPTIMUMS EXACTS")
    print("=" * 70)
    
    for scenario_name, res in results.items():
        if res['fitness'] > 0:
            print(f"\n{scenario_name}:")
            print(f"  Fitness max théorique: {res['fitness']:.2f}")
            print(f"  SUs servis max: {res['served_sus']}")
            print(f"  QoS max: {res['qos_rate']:.1f}%")
        else:
            print(f"\n{scenario_name}: Échec - {res['status']}")
    
    return results


if __name__ == "__main__":
    results = main()