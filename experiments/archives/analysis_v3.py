# analysis_v3.py
import pandas as pd
import matplotlib.pyplot as plt

# Lire le fichier généré
df = pd.read_csv("v3_only_scenario1_20260102_143000.csv")  # adaptez le nom

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(df["Run"], df["Fitness"], 'o-', color='blue')
axes[0].axhline(y=df["Fitness"].mean(), color='red', linestyle='--', label=f'mean={df["Fitness"].mean():.1f}')
axes[0].set_xlabel("Run")
axes[0].set_ylabel("Fitness")
axes[0].set_title(f"V3 Fitness ({SCENARIO})")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(df["Run"], df["Time"], 'o-', color='green')
axes[1].axhline(y=df["Time"].mean(), color='red', linestyle='--', label=f'mean={df["Time"].mean():.3f}s')
axes[1].set_xlabel("Run")
axes[1].set_ylabel("Time (s)")
axes[1].set_title(f"V3 Execution Time")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("v3_analysis.png", dpi=150)
plt.show()

# Boxplot comparison (si vous avez plusieurs versions)
print("\n📈 Distribution:")
print(df[["Fitness", "Time"]].describe())