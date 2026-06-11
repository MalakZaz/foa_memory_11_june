"""
analyze_results.py - Generate tables and figures from experimental data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create figures directory
os.makedirs('figures', exist_ok=True)

# Load data
df = pd.read_csv('results/experimental_data.csv')

print("=" * 60)
print("EXPERIMENTAL RESULTS ANALYSIS")
print("=" * 60)

# ============================================================
# 1. STATISTICS PER SCENARIO AND VERSION
# ============================================================

print("\n1. STATISTICS PER SCENARIO AND VERSION")
print("-" * 60)

stats = df.groupby(['Scenario', 'Version']).agg({
    'Fitness': ['mean', 'std'],
    'Time': ['mean', 'std'],
    'Served_SUs': ['mean', 'std'],
    'QoS_Rate': ['mean', 'std'],
    'Avoidance_Rate': ['mean', 'std']
}).round(2)

print(stats)

# ============================================================
# 2. GENERATE LATEX TABLE FOR RESULTS
# ============================================================

print("\n2. GENERATING LATEX TABLE...")

latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Performance comparison of FOA variants (30 runs)}
\\label{tab:results}
\\begin{tabular}{|l|c|c|c|c|}
\\hline
\\textbf{Version} & \\textbf{Fitness} & \\textbf{Time (s)} & \\textbf{Served SUs} & \\textbf{QoS (\\%)} \\\\
\\hline
"""

versions = ['V0', 'V1', 'V2', 'V3']
scenario_names = ['scenario1', 'scenario2', 'scenario3']
scenario_labels = ['20 SUs, 10 ch', '40 SUs, 15 ch', '60 SUs, 20 ch']

for i, scenario in enumerate(scenario_names):
    latex_table += f"\\multicolumn{{5}}{{|c|}}{{\\textbf{{{scenario_labels[i]}}}}} \\\\ \\hline\n"
    for version in versions:
        subset = df[(df['Scenario'] == scenario) & (df['Version'] == version)]
        if len(subset) > 0:
            fitness_mean = subset['Fitness'].mean()
            fitness_std = subset['Fitness'].std()
            time_mean = subset['Time'].mean()
            time_std = subset['Time'].std()
            served_mean = subset['Served_SUs'].mean()
            qos_mean = subset['QoS_Rate'].mean()
            
            latex_table += f"{version} & {fitness_mean:.2f} $\\pm$ {fitness_std:.2f} & "
            latex_table += f"{time_mean:.2f} $\\pm$ {time_std:.2f} & "
            latex_table += f"{served_mean:.1f} & {qos_mean:.1f}\\% \\\\ \\hline\n"

latex_table += """
\\end{tabular}
\\end{table}
"""

with open('results_table.tex', 'w') as f:
    f.write(latex_table)

print("✅ results_table.tex generated")

# ============================================================
# 3. GENERATE LATEX TABLE FOR PARAMETERS
# ============================================================

print("\n3. GENERATING PARAMETERS TABLE...")

params_table = """
\\begin{table}[htbp]
\\centering
\\caption{Simulation parameters}
\\label{tab:parameters}
\\begin{tabular}{|l|l|l|}
\\hline
\\textbf{Parameter} & \\textbf{Symbol} & \\textbf{Value} \\\\
\\hline
\\multicolumn{3}{|c|}{\\textbf{System parameters}} \\\\
\\hline
Number of Secondary Users & $N$ & 20, 40, 60 \\\\
Number of Channels & $M$ & 10, 15, 20 \\\\
Time window & $\\Delta t$ & $\\max(d_s)$ \\\\
Random seed & $\\text{seed}$ & 42 \\\\
\\hline
\\multicolumn{3}{|c|}{\\textbf{QoS parameters}} \\\\
\\hline
Minimum SINR threshold & $\\gamma_{\\min}$ & 2.0 \\\\
Minimum data rate & $R_{\\min}$ & 1.0 Mbps \\\\
Transmission power & $P_{\\text{TX}}$ & 1.0 W \\\\
Noise power & $\\sigma^2$ & 0.1 W \\\\
\\hline
\\multicolumn{3}{|c|}{\\textbf{FOA parameters}} \\\\
\\hline
Population size & $\\text{PopSize}$ & 30 \\\\
Maximum iterations & $\\text{MaxIter}$ & 100 \\\\
Number of runs & $\\text{Runs}$ & 30 \\\\
Exploration factor & $\\eta$ & 0.1 \\\\
\\hline
\\multicolumn{3}{|c|}{\\textbf{Niching parameters (V1, V3)}} \\\\
\\hline
Sharing radius & $\\sigma_{\\text{share}}$ & 0.5 \\\\
Sharing exponent & $\\alpha_{\\text{share}}$ & 1.0 \\\\
\\hline
\\multicolumn{3}{|c|}{\\textbf{Bloom memory parameters (V2, V3)}} \\\\
\\hline
Bloom filter size & $m$ & 1000 bits \\\\
Number of hash functions & $k$ & 5 \\\\
Novelty threshold & $\\theta$ & 0.5 \\\\
Bad solution threshold & $\\beta$ & 0.7 \\\\
Mutation rate & $p_{\\text{mut}}$ & 0.2 \\\\
\\hline
\\end{tabular}
\\end{table}
"""

with open('parameters_table.tex', 'w') as f:
    f.write(params_table)

print("✅ parameters_table.tex generated")

# ============================================================
# 4. GENERATE CONVERGENCE PLOTS (if history data available)
# ============================================================

print("\n4. GENERATING CONVERGENCE PLOTS...")

# Note: This requires history data. If not available, skip.
try:
    # Example: load history from separate file if exists
    # For now, create placeholder
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    versions_plot = ['V0', 'V1', 'V2', 'V3']
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, (ax, version, color) in enumerate(zip(axes.flat, versions_plot, colors)):
        subset = df[df['Version'] == version]
        fitness_by_run = subset.groupby('Run')['Fitness'].mean()
        
        ax.plot(fitness_by_run.values, color=color, linewidth=2)
        ax.set_xlabel('Run Number')
        ax.set_ylabel('Fitness')
        ax.set_title(f'{version} - Convergence')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/convergence_curves.png', dpi=300)
    plt.close()
    print("✅ convergence_curves.png generated")
except:
    print("⚠️ Convergence plots skipped (history data not available)")

# ============================================================
# 5. GENERATE BAR CHARTS
# ============================================================

print("\n5. GENERATING BAR CHARTS...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Fitness comparison
ax = axes[0]
for version in versions:
    subset = df[df['Version'] == version]
    fitness_mean = subset.groupby('Scenario')['Fitness'].mean()
    scenarios = ['S1', 'S2', 'S3']
    ax.bar([f'{version}_{s}' for s in scenarios], fitness_mean.values, label=version, alpha=0.7)
ax.set_xlabel('Scenario')
ax.set_ylabel('Fitness')
ax.set_title('Fitness Comparison')
ax.tick_params(axis='x', rotation=45)

# Time comparison
ax = axes[1]
for version in versions:
    subset = df[df['Version'] == version]
    time_mean = subset.groupby('Scenario')['Time'].mean()
    ax.bar([f'{version}_{s}' for s in scenarios], time_mean.values, alpha=0.7)
ax.set_xlabel('Scenario')
ax.set_ylabel('Time (s)')
ax.set_title('Execution Time Comparison')
ax.tick_params(axis='x', rotation=45)

# QoS comparison
ax = axes[2]
for version in versions:
    subset = df[df['Version'] == version]
    qos_mean = subset.groupby('Scenario')['QoS_Rate'].mean()
    ax.bar([f'{version}_{s}' for s in scenarios], qos_mean.values, alpha=0.7)
ax.set_xlabel('Scenario')
ax.set_ylabel('QoS Rate (%)')
ax.set_title('QoS Satisfaction Comparison')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/bar_charts.png', dpi=300)
plt.close()
print("✅ bar_charts.png generated")

# ============================================================
# 6. SUMMARY TABLE FOR CONSOLE
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY TABLE (Best Performance per Scenario)")
print("=" * 60)

for scenario in scenario_names:
    print(f"\n{scenario.upper()}:")
    subset = df[df['Scenario'] == scenario]
    best_fitness = subset.loc[subset.groupby('Version')['Fitness'].idxmax()]
    print(best_fitness[['Version', 'Fitness', 'Time', 'QoS_Rate']].to_string(index=False))

print("\n" + "=" * 60)
print("✅ Analysis complete!")
print("=" * 60)
print("\nGenerated files:")
print("  - results_table.tex")
print("  - parameters_table.tex")
print("  - figures/convergence_curves.png")
print("  - figures/bar_charts.png")